"""Tests for bounded rubric-model scheduling."""

from __future__ import annotations

import asyncio

import pytest

from hypotest.rubric_dispatcher import (
    RubricDispatchConfig,
    RubricDispatcher,
    RubricDispatchError,
    is_retryable_rubric_error,
)


def _config(**overrides) -> RubricDispatchConfig:
    values = {
        "enabled": True,
        "max_concurrency": 1,
        "max_outstanding": 4,
        "first_attempt_reserved_slots": 1,
        "retry_reserved_slots": 0,
        "attempt_timeout_seconds": 1.0,
        "ready_queue_timeout_seconds": 1.0,
        "logical_timeout_seconds": 2.0,
        "max_attempts": 2,
        "retry_backoff_initial_seconds": 0.01,
        "retry_backoff_max_seconds": 0.01,
    }
    return RubricDispatchConfig(**(values | overrides))


def test_config_rejects_impossible_reservations() -> None:
    with pytest.raises(ValueError, match="reserved slots"):
        _config(max_concurrency=2, first_attempt_reserved_slots=2, retry_reserved_slots=1)


def test_retry_classification_uses_status_codes() -> None:
    class ProviderError(Exception):
        status_code = 529

    class ClientError(Exception):
        status_code = 400

    assert is_retryable_rubric_error(ProviderError()) is True
    assert is_retryable_rubric_error(ClientError()) is False
    assert is_retryable_rubric_error(ValueError("bad rubric response")) is True


@pytest.mark.asyncio
async def test_dispatcher_bounds_physical_concurrency() -> None:
    dispatcher = RubricDispatcher(
        _config(
            max_concurrency=2,
            max_outstanding=4,
            first_attempt_reserved_slots=1,
            retry_reserved_slots=1,
            max_attempts=1,
        )
    )
    release = asyncio.Event()
    two_started = asyncio.Event()
    active = 0
    max_active = 0

    async def operation(_attempt: int) -> int:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            two_started.set()
        try:
            await release.wait()
            return active
        finally:
            active -= 1

    tasks = [asyncio.create_task(dispatcher.run(operation)) for _ in range(4)]
    await asyncio.wait_for(two_started.wait(), timeout=1)
    assert max_active == 2
    assert (await dispatcher.snapshot())["active_first"] == 2

    release.set()
    await asyncio.gather(*tasks)
    snapshot = await dispatcher.snapshot()
    assert snapshot["active_first"] == 0
    assert snapshot["outstanding"] == 0


@pytest.mark.asyncio
async def test_retry_backoff_does_not_hold_attempt_slot() -> None:
    dispatcher = RubricDispatcher(
        _config(
            retry_backoff_initial_seconds=0.1,
            retry_backoff_max_seconds=0.1,
        )
    )
    first_failed = asyncio.Event()
    order: list[str] = []

    async def retrying(attempt: int) -> str:
        await asyncio.sleep(0)
        order.append(f"retrying-{attempt}")
        if attempt == 1:
            first_failed.set()
            raise ValueError("retry")
        return "recovered"

    async def fresh(_attempt: int) -> str:
        await asyncio.sleep(0)
        order.append("fresh")
        return "fresh"

    retrying_task = asyncio.create_task(dispatcher.run(retrying))
    await asyncio.wait_for(first_failed.wait(), timeout=1)
    fresh_result = await asyncio.wait_for(dispatcher.run(fresh), timeout=1)
    retrying_result = await asyncio.wait_for(retrying_task, timeout=1)

    assert fresh_result.value == "fresh"
    assert retrying_result.value == "recovered"
    assert order == ["retrying-1", "fresh", "retrying-2"]
    assert retrying_result.metrics.retry_backoff_seconds_total == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_attempt_timeout_releases_slot_before_retry() -> None:
    dispatcher = RubricDispatcher(
        _config(
            attempt_timeout_seconds=0.02,
            retry_backoff_initial_seconds=0,
            retry_backoff_max_seconds=0,
        )
    )
    cancelled = asyncio.Event()

    async def operation(attempt: int) -> str:
        if attempt == 1:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return "ok"

    result = await asyncio.wait_for(dispatcher.run(operation), timeout=1)

    assert result.value == "ok"
    assert result.metrics.attempts == 2
    assert cancelled.is_set()
    snapshot = await dispatcher.snapshot()
    assert snapshot["active_first"] == 0
    assert snapshot["active_retry"] == 0


@pytest.mark.asyncio
async def test_retry_reservation_prevents_fresh_attempt_starvation() -> None:
    dispatcher = RubricDispatcher(
        _config(
            max_concurrency=2,
            max_outstanding=4,
            first_attempt_reserved_slots=1,
            retry_reserved_slots=1,
            retry_backoff_initial_seconds=0.05,
            retry_backoff_max_seconds=0.05,
        )
    )
    first_failed = asyncio.Event()
    blocker_b_started = asyncio.Event()
    blocker_c_started = asyncio.Event()
    retry_started = asyncio.Event()
    fresh_d_started = asyncio.Event()
    release_b = asyncio.Event()
    release_c = asyncio.Event()
    release_retry = asyncio.Event()

    async def blocker_b(_attempt: int) -> str:
        blocker_b_started.set()
        await release_b.wait()
        return "b"

    async def blocker_c(_attempt: int) -> str:
        blocker_c_started.set()
        await release_c.wait()
        return "c"

    async def retrying(attempt: int) -> str:
        if attempt == 1:
            first_failed.set()
            raise ValueError("retry")
        retry_started.set()
        await release_retry.wait()
        return "retry"

    async def fresh_d(_attempt: int) -> str:
        await asyncio.sleep(0)
        fresh_d_started.set()
        return "d"

    task_b = asyncio.create_task(dispatcher.run(blocker_b))
    await asyncio.wait_for(blocker_b_started.wait(), timeout=1)
    retry_task = asyncio.create_task(dispatcher.run(retrying))
    await asyncio.wait_for(first_failed.wait(), timeout=1)
    task_c = asyncio.create_task(dispatcher.run(blocker_c))
    await asyncio.wait_for(blocker_c_started.wait(), timeout=1)
    task_d = asyncio.create_task(dispatcher.run(fresh_d))
    await asyncio.sleep(0.06)

    release_c.set()
    await asyncio.wait_for(retry_started.wait(), timeout=1)
    assert fresh_d_started.is_set() is False

    release_retry.set()
    await asyncio.wait_for(fresh_d_started.wait(), timeout=1)
    release_b.set()
    await asyncio.gather(task_b, task_c, task_d, retry_task)


@pytest.mark.asyncio
async def test_ready_queue_timeout_is_terminal_and_releases_admission() -> None:
    dispatcher = RubricDispatcher(
        _config(
            max_outstanding=2,
            ready_queue_timeout_seconds=0.02,
            max_attempts=1,
        )
    )
    blocker_started = asyncio.Event()
    release = asyncio.Event()
    queued_called = False

    async def blocker(_attempt: int) -> str:
        blocker_started.set()
        await release.wait()
        return "done"

    async def queued(_attempt: int) -> str:
        nonlocal queued_called
        await asyncio.sleep(0)
        queued_called = True
        return "unexpected"

    blocker_task = asyncio.create_task(dispatcher.run(blocker))
    await asyncio.wait_for(blocker_started.wait(), timeout=1)
    with pytest.raises(RubricDispatchError) as error:
        await dispatcher.run(queued)
    assert error.value.reason == "ready_queue_timeout"
    assert error.value.metrics.first_attempt_queue_wait_seconds >= 0.02
    assert queued_called is False

    release.set()
    await blocker_task
    assert (await dispatcher.snapshot())["outstanding"] == 0


@pytest.mark.asyncio
async def test_capacity_excess_fails_without_hidden_waiter() -> None:
    dispatcher = RubricDispatcher(_config(max_outstanding=1, max_attempts=1))
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocker(_attempt: int) -> None:
        started.set()
        await release.wait()

    task = asyncio.create_task(dispatcher.run(blocker))
    await asyncio.wait_for(started.wait(), timeout=1)

    with pytest.raises(RubricDispatchError) as error:
        await dispatcher.run(blocker)
    assert error.value.reason == "capacity_exceeded"
    assert (await dispatcher.snapshot())["rejected"] == 1

    release.set()
    await task
