"""Bounded scheduling for rubric-model requests."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass
from typing import Any, Generic, Literal, TypeVar

import httpx
from litellm import APIConnectionError, InternalServerError, RateLimitError, ServiceUnavailableError, Timeout
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

T = TypeVar("T")
AttemptKind = Literal["first", "retry"]


class RubricDispatchConfig(BaseModel):
    """Queueing, timeout, and retry policy for one shared rubric endpoint."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_concurrency: int = Field(default=16, ge=1)
    max_outstanding: int = Field(default=512, ge=1)
    first_attempt_reserved_slots: int = Field(default=4, ge=0)
    retry_reserved_slots: int = Field(default=4, ge=0)
    attempt_timeout_seconds: float = Field(default=450.0, gt=0)
    ready_queue_timeout_seconds: float = Field(default=1800.0, gt=0)
    logical_timeout_seconds: float = Field(default=3600.0, gt=0)
    max_attempts: int = Field(default=4, ge=1)
    retry_backoff_initial_seconds: float = Field(default=10.0, ge=0)
    retry_backoff_max_seconds: float = Field(default=60.0, ge=0)
    retry_backoff_multiplier: float = Field(default=2.0, ge=1)

    @model_validator(mode="after")
    def validate_capacity(self) -> RubricDispatchConfig:
        if self.max_outstanding < self.max_concurrency:
            raise ValueError("rubric dispatch max_outstanding must be >= max_concurrency")
        if self.first_attempt_reserved_slots + self.retry_reserved_slots > self.max_concurrency:
            raise ValueError("rubric dispatch reserved slots must sum to <= max_concurrency")
        if self.retry_backoff_max_seconds < self.retry_backoff_initial_seconds:
            raise ValueError("rubric dispatch retry_backoff_max_seconds must be >= the initial backoff")
        return self


@dataclass(slots=True)
class RubricDispatchMetrics:
    """Per-logical-request scheduler accounting."""

    attempts: int = 0
    retries: int = 0
    queue_wait_seconds_total: float = 0.0
    queue_wait_seconds_max: float = 0.0
    first_attempt_queue_wait_seconds: float = 0.0
    attempt_seconds_total: float = 0.0
    attempt_seconds_max: float = 0.0
    retry_backoff_seconds_total: float = 0.0
    outstanding_at_submit: int = 0
    logical_seconds: float = 0.0
    final_reason: str = ""

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RubricDispatchResult(Generic[T]):
    value: T
    metrics: RubricDispatchMetrics


class RubricDispatchError(RuntimeError):
    """Terminal scheduler failure with structured accounting."""

    def __init__(
        self,
        reason: str,
        metrics: RubricDispatchMetrics,
        *,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(f"Rubric dispatch failed: {reason}")
        self.reason = reason
        self.metrics = metrics
        self.cause = cause


class RubricAttemptTimeoutError(TimeoutError):
    """A physical model request exceeded its local attempt budget."""


class _ReadyQueueTimeoutError(TimeoutError):
    def __init__(self, message: str = "", *, wait_seconds: float = 0.0) -> None:
        super().__init__(message)
        self.wait_seconds = wait_seconds


@dataclass(slots=True)
class _Slot:
    kind: AttemptKind
    released: bool = False


@dataclass(slots=True)
class _Waiter:
    kind: AttemptKind
    deadline: float
    sequence: int
    future: asyncio.Future[_Slot]
    assigned_slot: _Slot | None = None


def _exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(chain) < 8:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def is_retryable_rubric_error(error: BaseException) -> bool:
    """Classify transient provider and response-parse failures without text matching."""
    retryable_types = (
        ValueError,
        TimeoutError,
        ConnectionError,
        httpx.TransportError,
        Timeout,
        APIConnectionError,
        InternalServerError,
        RateLimitError,
        ServiceUnavailableError,
    )
    for item in _exception_chain(error):
        if isinstance(item, retryable_types):
            return True
        status_code = getattr(item, "status_code", None)
        if isinstance(status_code, int) and (status_code in {408, 425, 429, 529} or status_code >= 500):
            return True
    return False


class RubricDispatcher:
    """Schedule bounded physical attempts shared by all environments in a dataset."""

    def __init__(self, config: RubricDispatchConfig):
        if not config.enabled:
            raise ValueError("RubricDispatcher requires enabled=true")
        self.config = config
        self._lock = asyncio.Lock()
        self._first_ready: list[tuple[float, int, _Waiter]] = []
        self._retry_ready: list[tuple[float, int, _Waiter]] = []
        self._sequence = itertools.count()
        self._active: dict[AttemptKind, int] = {"first": 0, "retry": 0}
        self._outstanding = 0
        self._backing_off = 0
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        self._max_first_ready = 0
        self._max_retry_ready = 0
        self._last_flexible_kind: AttemptKind = "retry"

    async def run(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        operation: Callable[[int], Coroutine[Any, Any, T]],
        *,
        retryable: Callable[[BaseException], bool] = is_retryable_rubric_error,
    ) -> RubricDispatchResult[T]:
        """Run a logical rubric request as bounded, independently scheduled attempts."""
        loop = asyncio.get_running_loop()
        submitted_at = loop.time()
        logical_deadline = submitted_at + self.config.logical_timeout_seconds
        metrics = RubricDispatchMetrics()

        async with self._lock:
            metrics.outstanding_at_submit = self._outstanding
            if self._outstanding >= self.config.max_outstanding:
                self._rejected += 1
                metrics.final_reason = "capacity_exceeded"
                raise RubricDispatchError("capacity_exceeded", metrics)
            self._outstanding += 1

        final_reason = "cancelled"
        try:
            last_error: BaseException | None = None
            for attempt_number in range(1, self.config.max_attempts + 1):
                if attempt_number > 1:
                    delay = self._retry_delay(attempt_number)
                    if loop.time() + delay >= logical_deadline:
                        final_reason = "logical_deadline"
                        break
                    metrics.retries += 1
                    metrics.retry_backoff_seconds_total += delay
                    await self._sleep_backoff(delay)

                if loop.time() >= logical_deadline:
                    final_reason = "logical_deadline"
                    break

                kind: AttemptKind = "first" if attempt_number == 1 else "retry"
                try:
                    slot, queue_wait = await self._acquire_slot(kind, logical_deadline)
                except _ReadyQueueTimeoutError as queue_error:
                    metrics.queue_wait_seconds_total += queue_error.wait_seconds
                    metrics.queue_wait_seconds_max = max(metrics.queue_wait_seconds_max, queue_error.wait_seconds)
                    if attempt_number == 1:
                        metrics.first_attempt_queue_wait_seconds = queue_error.wait_seconds
                    last_error = queue_error
                    final_reason = "ready_queue_timeout"
                    break

                metrics.queue_wait_seconds_total += queue_wait
                metrics.queue_wait_seconds_max = max(metrics.queue_wait_seconds_max, queue_wait)
                if attempt_number == 1:
                    metrics.first_attempt_queue_wait_seconds = queue_wait

                attempt_started = loop.time()
                metrics.attempts += 1
                attempt_error: BaseException | None = None
                remaining = logical_deadline - attempt_started
                try:
                    if remaining <= 0:
                        attempt_error = RubricAttemptTimeoutError(
                            "logical request deadline elapsed before provider attempt"
                        )
                    else:
                        attempt_budget = min(self.config.attempt_timeout_seconds, remaining)
                        value = await self._run_attempt(operation, attempt_number, attempt_budget)
                except Exception as error:
                    attempt_error = error
                finally:
                    attempt_seconds = loop.time() - attempt_started
                    metrics.attempt_seconds_total += attempt_seconds
                    metrics.attempt_seconds_max = max(metrics.attempt_seconds_max, attempt_seconds)
                    await self._release_slot(slot)

                if attempt_error is None:
                    metrics.logical_seconds = loop.time() - submitted_at
                    metrics.final_reason = "success"
                    final_reason = "success"
                    return RubricDispatchResult(value=value, metrics=metrics)

                last_error = attempt_error
                if not retryable(attempt_error):
                    final_reason = "non_retryable_error"
                    break
                if attempt_number == self.config.max_attempts:
                    final_reason = "attempts_exhausted"

            metrics.logical_seconds = loop.time() - submitted_at
            metrics.final_reason = final_reason
            raise RubricDispatchError(final_reason, metrics, cause=last_error) from last_error
        finally:
            async with self._lock:
                self._outstanding -= 1
                if final_reason == "success":
                    self._completed += 1
                elif final_reason != "cancelled":
                    self._failed += 1
                self._maybe_log_stats_locked()

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            return self._snapshot_locked()

    async def _run_attempt(
        self,
        operation: Callable[[int], Coroutine[Any, Any, T]],
        attempt_number: int,
        timeout_seconds: float,
    ) -> T:
        task: asyncio.Task[T] = asyncio.create_task(operation(attempt_number))
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
            if not done:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise RubricAttemptTimeoutError(f"rubric attempt exceeded {timeout_seconds:.3f}s")
            return task.result()
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _sleep_backoff(self, delay: float) -> None:
        async with self._lock:
            self._backing_off += 1
        try:
            await asyncio.sleep(delay)
        finally:
            async with self._lock:
                self._backing_off -= 1

    async def _acquire_slot(self, kind: AttemptKind, logical_deadline: float) -> tuple[_Slot, float]:
        loop = asyncio.get_running_loop()
        ready_at = loop.time()
        dispatch_deadline = min(logical_deadline, ready_at + self.config.ready_queue_timeout_seconds)
        future: asyncio.Future[_Slot] = loop.create_future()
        waiter = _Waiter(
            kind=kind,
            deadline=dispatch_deadline,
            sequence=next(self._sequence),
            future=future,
        )

        async with self._lock:
            queue = self._queue(kind)
            heapq.heappush(queue, (waiter.deadline, waiter.sequence, waiter))
            self._max_first_ready = max(self._max_first_ready, len(self._first_ready))
            self._max_retry_ready = max(self._max_retry_ready, len(self._retry_ready))
            self._dispatch_locked(loop.time())

        try:
            timeout_seconds = max(0.0, dispatch_deadline - loop.time())
            async with asyncio.timeout(timeout_seconds):
                slot = await future
            return slot, loop.time() - ready_at
        except (TimeoutError, _ReadyQueueTimeoutError) as error:
            await self._cancel_waiter(waiter)
            wait_seconds = loop.time() - ready_at
            raise _ReadyQueueTimeoutError(
                f"{kind} attempt was not dispatched before its queue deadline",
                wait_seconds=wait_seconds,
            ) from error
        except asyncio.CancelledError:
            await self._cancel_waiter(waiter)
            raise

    async def _cancel_waiter(self, waiter: _Waiter) -> None:
        async with self._lock:
            if waiter.assigned_slot is not None:
                self._release_slot_locked(waiter.assigned_slot)
            elif not waiter.future.done():
                waiter.future.cancel()
            self._dispatch_locked(asyncio.get_running_loop().time())

    async def _release_slot(self, slot: _Slot) -> None:
        async with self._lock:
            self._release_slot_locked(slot)
            self._dispatch_locked(asyncio.get_running_loop().time())

    def _release_slot_locked(self, slot: _Slot) -> None:
        if slot.released:
            return
        slot.released = True
        self._active[slot.kind] -= 1

    def _dispatch_locked(self, now: float) -> None:
        while sum(self._active.values()) < self.config.max_concurrency:
            kind = self._select_kind_locked(now)
            if kind is None:
                return
            waiter = self._pop_waiter_locked(kind, now)
            if waiter is None:
                continue
            slot = _Slot(kind=kind)
            waiter.assigned_slot = slot
            self._active[kind] += 1
            if not waiter.future.done():
                waiter.future.set_result(slot)
            else:
                self._release_slot_locked(slot)

    def _select_kind_locked(self, now: float) -> AttemptKind | None:
        first = self._peek_waiter_locked("first", now)
        retry = self._peek_waiter_locked("retry", now)
        if first is None:
            return "retry" if retry is not None else None
        if retry is None:
            return "first"
        if self._active["first"] < self.config.first_attempt_reserved_slots:
            return "first"
        if self._active["retry"] < self.config.retry_reserved_slots:
            return "retry"
        if first.deadline < retry.deadline:
            choice: AttemptKind = "first"
        elif retry.deadline < first.deadline:
            choice = "retry"
        else:
            choice = "first" if self._last_flexible_kind == "retry" else "retry"
        self._last_flexible_kind = choice
        return choice

    def _peek_waiter_locked(self, kind: AttemptKind, now: float) -> _Waiter | None:
        queue = self._queue(kind)
        while queue:
            _, _, waiter = queue[0]
            if waiter.future.done():
                heapq.heappop(queue)
                continue
            if waiter.deadline <= now:
                heapq.heappop(queue)
                waiter.future.set_exception(_ReadyQueueTimeoutError())
                continue
            return waiter
        return None

    def _pop_waiter_locked(self, kind: AttemptKind, now: float) -> _Waiter | None:
        waiter = self._peek_waiter_locked(kind, now)
        if waiter is None:
            return None
        heapq.heappop(self._queue(kind))
        return waiter

    def _queue(self, kind: AttemptKind) -> list[tuple[float, int, _Waiter]]:
        return self._first_ready if kind == "first" else self._retry_ready

    def _retry_delay(self, attempt_number: int) -> float:
        exponent = max(0, attempt_number - 2)
        return min(
            self.config.retry_backoff_initial_seconds * self.config.retry_backoff_multiplier**exponent,
            self.config.retry_backoff_max_seconds,
        )

    def _snapshot_locked(self) -> dict[str, int]:
        return {
            "outstanding": self._outstanding,
            "active_first": self._active["first"],
            "active_retry": self._active["retry"],
            "ready_first": sum(not item[2].future.done() for item in self._first_ready),
            "ready_retry": sum(not item[2].future.done() for item in self._retry_ready),
            "backing_off": self._backing_off,
            "completed": self._completed,
            "failed": self._failed,
            "rejected": self._rejected,
            "max_ready_first": self._max_first_ready,
            "max_ready_retry": self._max_retry_ready,
        }

    def _maybe_log_stats_locked(self) -> None:
        finished = self._completed + self._failed
        if finished and finished % 128 == 0:
            logger.info("Rubric dispatcher stats: %s", self._snapshot_locked())
