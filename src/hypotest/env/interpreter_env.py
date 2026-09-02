"""InterpreterEnv: Standalone code execution environment for data analysis.

This module provides a lightweight, execution-focused environment for running
code in Jupyter kernels. It focuses on direct code execution via run_cell().
"""

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import random
import shutil
import time
import uuid
import warnings
from collections.abc import Mapping
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Literal, cast
from uuid import UUID

import httpx
import nbformat
import numpy as np
import tenacity
from aviary.core import (
    EnvStateMessage,
    Frame,
    Message,
    Messages,
    Tool,
    ToolCall,
    ToolRequestMessage,
)
from aviary.env import Environment
from lmi import LiteLLMModel
from nbformat import NotebookNode
from pydantic import BaseModel, Field, JsonValue, model_validator

from hypotest.rubric_dispatcher import RubricDispatcher, RubricDispatchError
from hypotest.rubric_provider_debug import RubricProviderDebugLogger

from . import config as cfg
from .code_safety import check_code_safety
from .config import ExecutionConfig
from .determinism import EnvSeeds
from .hybrid_gate import (
    HYBRID_GATE_PROMPT,
    hybrid_reward,
    parse_hybrid_response,
    synthesize_per_item_awards,
)
from .interpreter import ExecutionResult
from .prompts import (
    CORRECT_MSG,
    FAITHFULNESS_GATE_PROMPT,
    HYPOTHESIS_TASK_DESC,
    INCORRECT_MSG,
    RUBRIC_SCORE_PROMPT,
    PromptingConfig,
)
from .sandbox import (
    CapsuleRef,
    K8sFallbackScheduler,
    K8sSandboxSpec,
    OpenSandboxFallbackScheduler,
    OpenSandboxSpec,
    ResourceSpec,
    Sandbox,
    SandboxConfig,
    SandboxScheduler,
    make_sandbox,
)
from .step_context import ModelTurn, model_turns_from_action_info
from .tools.filesystem import FilesystemTool
from .utils import NBLanguage, render_notebook_for_rubric, view_notebook
from .wager import (
    WAGER_BETA_DEFAULT,
    WAGER_GAMMA_DEFAULT,
    clamp_confidence,
    score_with_wager,
)

RAY_INSTALLED = True
try:
    from ray.exceptions import RayActorError
except ImportError:
    RAY_INSTALLED = False

# Cell-replay budget for session recovery (swap + replay on backend death).
_REPLAY_BUDGET = int(os.getenv("SANDBOX_REPLAY_BUDGET", "50"))

# Transport failures that warrant a sandbox swap+replay — the kernel/pod/actor DIED.
# A cell timeout is NOT here: it comes back as an ExecutionResult (never an exception), and a
# slow-but-alive call shouldn't trigger a disruptive recovery. Connection loss + actor death only.
_RECOVERABLE_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)
if RAY_INSTALLED:
    _RECOVERABLE_TRANSPORT_ERRORS = (*_RECOVERABLE_TRANSPORT_ERRORS, RayActorError)

logger = logging.getLogger(__name__)


_warned_unsafe_execution: set[str] = set()
_BACKGROUND_CLEANUP_TASKS: set[asyncio.Task[None]] = set()


def _make_cleanup_path(path: Path) -> Path:
    return path.with_name(f".cleanup-{path.name}-{uuid.uuid4().hex}")


def _detach_dir_for_cleanup(path: Path) -> Path | None:
    if not path.exists():
        return None

    cleanup_path = _make_cleanup_path(path)
    while cleanup_path.exists():
        cleanup_path = _make_cleanup_path(path)

    path.replace(cleanup_path)
    return cleanup_path


def _schedule_dir_cleanup(path: Path) -> None:
    async def _cleanup() -> None:
        try:
            await asyncio.to_thread(shutil.rmtree, path, ignore_errors=True)
        except Exception:
            logger.warning("Background cleanup failed for %s", path, exc_info=True)

    task = asyncio.create_task(_cleanup())
    _BACKGROUND_CLEANUP_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_CLEANUP_TASKS.discard)


def _validate_generation_measurements(
    latency: cfg.GenerationLatencyConfig | None,
    turns: list[ModelTurn],
) -> None:
    """Require the source data selected by the generation accounting mode."""
    if isinstance(latency, cfg.TokenThroughputGenerationLatencyConfig):
        if not turns:
            raise ValueError("token_throughput generation latency requires NeMo Gym model-turn metadata")
        missing_usage = [turn.response_id for turn in turns if turn.usage is None]
        if missing_usage:
            raise ValueError(
                "token_throughput generation latency requires output-token usage for responses: "
                + ", ".join(missing_usage)
            )
    elif isinstance(latency, cfg.ReportedGenerationLatencyConfig):
        if not turns:
            raise ValueError("reported generation latency requires NeMo Gym model-turn metadata")
        missing_duration = [turn.response_id for turn in turns if turn.generation_seconds is None]
        if missing_duration:
            raise ValueError(
                "reported generation latency requires generation_seconds for responses: "
                + ", ".join(missing_duration)
            )


def _generation_seconds(
    latency: cfg.GenerationLatencyConfig | None,
    turns: list[ModelTurn | None],
    output_tokens: int,
) -> float:
    """Compute one step's configured generation charge from validated turns."""
    if isinstance(latency, cfg.GenerationLatencyEstimateConfig):
        return float(latency.seconds_per_generation) * len(turns)
    if isinstance(latency, cfg.TokenThroughputGenerationLatencyConfig):
        return output_tokens / float(latency.output_tokens_per_second)
    if isinstance(latency, cfg.ReportedGenerationLatencyConfig):
        return sum(
            turn.generation_seconds
            for turn in turns
            if turn is not None and turn.generation_seconds is not None
        )
    return 0.0


class ProblemInstance(BaseModel):
    id: UUID
    hypothesis: str
    protocol: str
    accepted: bool = Field(alias="answer")
    rubric: str
    max_score: int = Field(alias="max_points")
    input_data_path: str = ""
    faithfulness_rubric: str = ""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    nb_primary_language: str = Field(default=str(NBLanguage.PYTHON))

    @model_validator(mode="before")
    @classmethod
    def handle_language(cls, data: dict) -> dict:
        if data.get("nb_primary_language") is None:
            data["nb_primary_language"] = str(NBLanguage.PYTHON)
        return data


class InterpreterEnvState:
    """State container for the InterpreterEnv.

    Manages the kernel, notebook state, and execution tracking.
    Supports both local kernel execution and Docker-based execution.
    """

    def __init__(
        self,
        work_dir: Path,
        language: NBLanguage,
        execution_timeout: int = 600,
        timeout_recovery: Literal["none", "interrupt"] = "none",
        interrupt_grace_seconds: float = 10.0,
        safe_execute: bool = True,
        use_host_env_vars: bool = False,
        extra_envs: dict[str, str] | None = None,
        use_docker: bool = cfg.USE_DOCKER,
        use_enroot: bool = False,
        use_ray: bool = True,
        container_sqsh_path: Path | None = None,
        save_dir: Path | None = None,
        sandbox_memory_request_mb: int | None = None,
        sandbox_memory_limit_mb: int | None = None,
        sandbox_max_pids: int | None = None,
        sandbox_cpu_request: float | None = None,
        sandbox_cpu: float | None = None,
        sandbox_ephemeral_storage_gib: int | None = None,
        sandbox_gpu_count: int | None = None,
        sandbox_gpu_type: str | None = None,
        k8s_specs: list[K8sSandboxSpec] | None = None,
        opensandbox_spec: OpenSandboxSpec | None = None,
        sandbox_job_id: str | None = None,
        enable_recovery: bool = False,
        capsule_ref: CapsuleRef | None = None,
        seed: int | None = None,
        scheduler_seed: int | None = None,
    ):
        self.work_dir = work_dir
        self.language = language
        self.execution_timeout = execution_timeout
        self.safe_execute = safe_execute
        self.total_reward = 0.0
        self.use_host_env_vars = use_host_env_vars
        self.extra_envs = extra_envs or {}
        self.answer: str | None = None
        self.actions: list[str] = []
        self.done = False
        self.save_dir = save_dir

        # One execution backend behind a uniform interface — the state never
        # branches on backend type. make_sandbox interprets the use_docker /
        # use_enroot / use_ray selectors exactly once.
        self._sandbox_config = SandboxConfig(
            work_dir=work_dir,
            language=language,
            execution_timeout=execution_timeout,
            timeout_recovery=timeout_recovery,
            interrupt_grace_seconds=interrupt_grace_seconds,
            safe_execute=safe_execute,
            use_host_env_vars=use_host_env_vars,
            extra_envs=self.extra_envs,
            container_sqsh_path=container_sqsh_path,
            resources=ResourceSpec(
                mem_mb=sandbox_memory_limit_mb,
                mem_request_mb=sandbox_memory_request_mb,
                max_pids=sandbox_max_pids,
                cpu=sandbox_cpu,
                cpu_request=sandbox_cpu_request,
                disk_gib=sandbox_ephemeral_storage_gib,
                gpu=sandbox_gpu_count,
                gpu_type=sandbox_gpu_type,
            ),
            # Capsule identity for remote delivery. OpenSandbox supplies it as
            # init env; k8s posts it to /load_capsule. Local backends ignore it.
            ref=capsule_ref or CapsuleRef(),
            job_id=sandbox_job_id,
            use_docker=use_docker,
            use_enroot=use_enroot,
            use_ray=use_ray if RAY_INSTALLED else False,
            seed=seed,
        )
        self.sandbox: Sandbox = make_sandbox(self._sandbox_config)
        self._started = False
        # Optional remote placement, falling back to the configured local
        # backend. When set, start()/recover() acquire through the scheduler.
        # K8sSandbox ignores the use_* selectors and the fallback goes through make_sandbox, so both
        # arms share self._sandbox_config. Recovery stays dark-launched behind enable_recovery.
        scheduler_rng = random.Random(scheduler_seed) if scheduler_seed is not None else None
        if k8s_specs and opensandbox_spec is not None:
            raise ValueError("Configure either k8s_specs or opensandbox_spec, not both")
        self._scheduler: SandboxScheduler | None = None
        if k8s_specs:
            self._scheduler = K8sFallbackScheduler(
                self._sandbox_config,
                k8s_specs,
                self._sandbox_config,
                rng=scheduler_rng,
            )
        elif opensandbox_spec is not None:
            self._scheduler = OpenSandboxFallbackScheduler(
                self._sandbox_config,
                opensandbox_spec,
                self._sandbox_config,
            )
        self._enable_recovery = enable_recovery
        self._recovering = False

        # Initialize notebook structure for state tracking
        self.nb: NotebookNode = nbformat.v4.new_notebook()
        self.nb.metadata.kernelspec = language.make_kernelspec()
        self.notebook_runtime_errors: list[str] = []
        self._execution_count = 0

        self.raw_score: int = 0
        self.score: float = 0.0
        self.score_metadata: dict[str, Any] = {}
        self.rubric_model_raw_response: str = ""
        self.rubric_model_failed: bool = False
        self.rubric_model_fail_type: str = ""
        self.rubric_model_error_type: str = ""
        self.zero_reward: bool = False
        self.faithfulness_passed: bool | None = None
        self.faithfulness_metadata: dict[str, Any] = {}
        self.rubric_reward_raw: float = 0.0
        self.hybrid_reward_value: float = 0.0
        self.hybrid_metadata: dict[str, Any] = {}
        self.wager: float = 0.0
        self.wager_reward_shadow: float = 0.0
        self.wager_metadata: dict[str, Any] = {}
        self.cell_timeout_override_requests: list[float] = []

    async def start(self):
        """Start the execution backend (via the scheduler if one is configured)."""
        if self._scheduler is not None:
            self.sandbox = await self._scheduler.acquire(self._sandbox_config.ref, self._sandbox_config.resources)
        else:
            await self.sandbox.start()
        self._started = True

    async def _execute_with_recovery(self, code: str, timeout: float | None, req_uuid: str) -> ExecutionResult:  # noqa: ASYNC109
        """Execute via the sandbox; on a transport failure (not a cell timeout), swap+replay and retry once.

        Surfaces the recovery to the agent by prepending a notice to the returned
        ExecutionResult, so it knows its kernel state was rebuilt (and possibly clipped).
        Episode-time accounting happens above this method, once per ``req_uuid``;
        individual attempts and recovery replay are never charged independently.
        """
        try:
            return await self.sandbox.execute(code, timeout, req_uuid=req_uuid)
        except _RECOVERABLE_TRANSPORT_ERRORS as e:
            if not self._enable_recovery or self._recovering:
                raise
            logger.warning("sandbox transport failure (%s); recovering session", type(e).__name__)
            cells_before = len(self.nb.cells)
            recovered_len = await self.recover()
            result = await self.sandbox.execute(code, timeout, req_uuid=req_uuid)

            dropped = cells_before - recovered_len
            if dropped > 0:
                text = (
                    f"[session recovered after a sandbox failure: replayed {recovered_len} cell(s); "
                    f"{dropped} later cell(s) exceeded the replay budget and were dropped — the notebook "
                    f"was clipped to {recovered_len} cells, so earlier state is restored but later cells are gone]"
                )
            else:
                text = f"[session recovered after a sandbox failure: replayed {recovered_len} cell(s); state restored]"
            notice = nbformat.v4.new_output(output_type="stream", name="stderr", text=text + "\n")
            result.notebook_outputs = [notice, *result.notebook_outputs]
            return result

    async def recover(self) -> int:
        """Swap in a fresh sandbox, replay the cell history, and return the recovered cell count.

        Replay is capped at SANDBOX_REPLAY_BUDGET. If fewer cells are replayed than the notebook
        held, the notebook is CLIPPED to the recovered length (and `_execution_count` adjusted) so
        it stays consistent with the rebuilt kernel — the dropped cells' state is gone. Best-effort:
        replay reconstructs deterministic state only. Seeded RNG state is reproducible in deterministic
        mode; wall-clock values, external I/O, and other non-idempotent effects are not faithfully restored.
        """
        self._recovering = True
        try:
            with contextlib.suppress(Exception):
                await self.sandbox.close()
            if self._scheduler is not None:
                self.sandbox = await self._scheduler.acquire(self._sandbox_config.ref, self._sandbox_config.resources)
            else:
                self.sandbox = make_sandbox(self._sandbox_config)
                await self.sandbox.start()
            self._started = True

            original_len = len(self.nb.cells)
            replay_cells = self.nb.cells[:_REPLAY_BUDGET]
            for cell in replay_cells:
                if cell.get("cell_type") != "code":
                    continue
                with contextlib.suppress(Exception):
                    await self.sandbox.execute(cell.source)

            recovered_len = len(replay_cells)
            if recovered_len < original_len:
                # The rebuilt kernel only reflects the replayed cells — clip the notebook to match.
                self.nb.cells = self.nb.cells[:recovered_len]
                self._execution_count = recovered_len
                logger.warning(
                    "recovery dropped %d cell(s): notebook clipped %d -> %d (replay budget %d)",
                    original_len - recovered_len,
                    original_len,
                    recovered_len,
                    _REPLAY_BUDGET,
                )
            else:
                logger.warning("recovered session: replayed %d cell(s)", recovered_len)
            return recovered_len
        finally:
            self._recovering = False

    async def close(self):
        """Save the notebook, tear down the sandbox, and relocate the workspace."""
        nbformat.write(self.nb, self.work_dir / "notebook.ipynb")

        await self.sandbox.close()
        self._started = False

        if self.save_dir is not None and self.work_dir.exists():
            self.save_dir.parent.mkdir(parents=True, exist_ok=True)
            if self.save_dir.exists():
                try:
                    cleanup_path = _detach_dir_for_cleanup(self.save_dir)
                except Exception as e:
                    logger.warning("Failed to detach existing save_dir %s for cleanup: %s", self.save_dir, e)
                else:
                    if cleanup_path is not None:
                        logger.warning(
                            "Detached existing save_dir %s to %s for background cleanup", self.save_dir, cleanup_path
                        )
                        _schedule_dir_cleanup(cleanup_path)
            try:
                self.work_dir.replace(self.save_dir)
            except Exception as e:
                logger.warning("Failed to move work_dir %s to save_dir %s: %s", self.work_dir, self.save_dir, e)
        elif self.work_dir.exists():
            try:
                cleanup_path = _detach_dir_for_cleanup(self.work_dir)
            except Exception as e:
                logger.warning("Failed to detach workspace %s for background cleanup: %s", self.work_dir, e)
            else:
                if cleanup_path is not None:
                    logger.warning("Detached workspace %s to %s for background cleanup", self.work_dir, cleanup_path)
                    _schedule_dir_cleanup(cleanup_path)

    def _add_cell(self, code: str, result: "ExecutionResult") -> int:
        """Add a new code cell to the notebook with execution results.

        Args:
            code: The code that was executed
            result: The execution result

        Returns:
            The cell index of the added cell
        """
        self._execution_count += 1

        cell = nbformat.v4.new_code_cell(
            source=code,
            outputs=result.notebook_outputs,
            execution_count=self._execution_count,
        )

        self.nb.cells.append(cell)
        cell_idx = len(self.nb.cells) - 1

        # Track errors if any
        if result.error_occurred:
            error_msg = result.get_error_message()
            if error_msg:
                self.notebook_runtime_errors.append(f"Cell {self._execution_count}: {error_msg}")

        return cell_idx

    def _update_cell(self, idx: int, code: str, result: "ExecutionResult") -> None:
        """Update an existing cell's source and outputs.

        Args:
            idx: The cell index to update
            code: The new code
            result: The execution result
        """
        cell = self.nb.cells[idx]
        cell.source = code
        cell.outputs = result.notebook_outputs

        # Update error tracking - remove old error for this cell, add new if any
        if result.error_occurred:
            error_msg = result.get_error_message()
            if error_msg:
                # Remove any existing error for this cell
                self.notebook_runtime_errors = [
                    err for err in self.notebook_runtime_errors if not err.startswith(f"Cell {idx + 1}:")
                ]
                self.notebook_runtime_errors.append(f"Cell {idx + 1}: {error_msg}")

    def get_execution_summary(self) -> dict[str, Any]:
        """Summary of execution history + current state (backend-agnostic)."""
        error_count = len(self.notebook_runtime_errors)
        recent_errors = self.notebook_runtime_errors[-3:] if self.notebook_runtime_errors else []

        return {
            "total_executions": self._execution_count,
            "error_count": error_count,
            "recent_errors": recent_errors,
            "last_execution": None,
            "is_ready": self._started,
            "language": self.language.value,
            "work_dir": str(self.work_dir),
        }

    async def execute_and_add_cell(
        self,
        code: str,
        cell_idx: int | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
        req_uuid: str = "",
    ) -> tuple[ExecutionResult, int]:
        """Execute code and atomically update notebook.

        Args:
            code: Code to execute
            cell_idx: Cell index to update (None = append new cell)
            timeout: Optional execution timeout

        Returns:
            Tuple of (ExecutionResult, actual_cell_index)
        """
        if self.safe_execute:
            block_reason = check_code_safety(code, self.language)
            if block_reason is not None:
                logger.warning("Blocked code execution in execute_and_add_cell: %s", block_reason)
                error_output = nbformat.v4.new_output(
                    output_type="error",
                    ename="SecurityError",
                    evalue=block_reason,
                    traceback=[f"SecurityError: {block_reason}"],
                )
                result = ExecutionResult(
                    notebook_outputs=[error_output],
                    error_occurred=True,
                    execution_time=0.0,
                )
                if cell_idx is None or cell_idx >= len(self.nb.cells):
                    actual_idx = self._add_cell(code, result)
                else:
                    self._update_cell(cell_idx, code, result)
                    actual_idx = cell_idx
                return result, actual_idx
        elif "unsafe_execution" not in _warned_unsafe_execution:
            logger.warning(
                "Running code sandbox without safety filter, may result in destructive code running on the node"
            )
            _warned_unsafe_execution.add("unsafe_execution")

        result = await self._execute_with_recovery(code, timeout, req_uuid)

        if cell_idx is None or cell_idx >= len(self.nb.cells):
            actual_idx = self._add_cell(code, result)
        else:
            self._update_cell(cell_idx, code, result)
            actual_idx = cell_idx

        return result, actual_idx


class InterpreterEnvConfig(BaseModel):
    """Configuration for preparing the InterpreterEnv during task creation."""

    language: NBLanguage = Field(default=NBLanguage.PYTHON)
    prompting_config: PromptingConfig = Field(default_factory=PromptingConfig)
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    max_steps: int = cfg.AGENT_MAX_STEPS
    use_ray: bool = False
    use_docker: bool = cfg.USE_DOCKER
    use_enroot: bool = False
    container_sqsh_path: Path | None = None
    normalize_reward: bool = True
    enable_faithfulness_gate: bool = False
    faithfulness_mode: Literal["off", "binary", "shadow", "hybrid"] = "off"
    wager_mode: Literal["off", "shadow", "active"] = "off"
    wager_beta: float = WAGER_BETA_DEFAULT
    wager_gamma: float = WAGER_GAMMA_DEFAULT
    cell_timeout_override_mode: Literal["off", "on"] = "off"
    cell_timeout_min: float = 60.0
    cell_timeout_max: float = 1200.0
    replace_image_payloads_with_placeholders: bool = True
    include_images_in_rubric_model: bool = True
    max_rubric_images: int = 20
    rubric_notebook_serialization: Literal["auto", "multimodal", "legacy"] = "auto"
    # Session recovery (swap a fresh sandbox + replay the cell history on a transport failure);
    # dark-launched, opt-in. See InterpreterEnvState.recover().
    enable_recovery: bool = False
    # Opt-in k8s (agent-sandbox) placement: each spec is a warmpool/template target the scheduler
    # load-balances across, falling back to the configured (enroot) backend. Empty = disabled.
    k8s_sandbox_specs: list[K8sSandboxSpec] = Field(default_factory=list)
    # Raw OpenSandbox SDK placement. The remote container runs the same kernel
    # HTTP protocol and falls back to the locally staged backend when allocation
    # or reachability fails.
    opensandbox_spec: OpenSandboxSpec | None = None
    # Opaque job identity stamped on k8s claims (hashed) for the clean-on-startup sweep; flows to
    # SandboxConfig.job_id. None => claims are unattributed (sweep is a no-op).
    sandbox_job_id: str | None = None
    # Remote capsule delivery: OpenSandbox pulls during container init and k8s
    # uses /load_capsule. Off for pure-exec/no-data smoke tests. Local fallback
    # still receives the separately staged work_dir.
    pull_capsule_in_pod: bool = True
    # `seed` is the dataset-level base seed. Component seeds are derived from it
    # and env_idx without consuming a mutable RNG, so streams remain independent.
    deterministic: bool = False
    seed: int = 0
    env_idx: int = 0
    rubric_seed: int | None = None

    @model_validator(mode="after")
    def _migrate_enable_faithfulness_gate(self) -> "InterpreterEnvConfig":
        if self.enable_faithfulness_gate and self.faithfulness_mode == "off":
            warnings.warn(
                "enable_faithfulness_gate=True is deprecated; use faithfulness_mode='binary' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.faithfulness_mode = "binary"
        return self

    @model_validator(mode="after")
    def _validate_wager_requires_gate(self) -> "InterpreterEnvConfig":
        if self.wager_mode != "off" and self.faithfulness_mode == "off":
            raise ValueError(
                f"wager_mode={self.wager_mode!r} requires faithfulness_mode "
                "∈ {'shadow', 'hybrid'}; got 'off'. Wager uses the gate's "
                "correct signal; it cannot operate standalone."
            )
        return self

    @model_validator(mode="after")
    def _validate_one_remote_backend(self) -> "InterpreterEnvConfig":
        if self.k8s_sandbox_specs and self.opensandbox_spec is not None:
            raise ValueError("Configure either k8s_sandbox_specs or opensandbox_spec, not both")
        return self


class InterpreterEnv(Environment[InterpreterEnvState]):
    """Standalone environment for code execution and data analysis.

    This environment provides direct code execution via run_cell() without
    requiring notebook file I/O. It maintains an in-memory notebook for
    trajectory tracking and state export.
    """

    def __init__(
        self,
        *,
        problem: ProblemInstance,
        work_dir: Path,
        rubric_model: LiteLLMModel | None = None,
        rubric_dispatcher: RubricDispatcher | None = None,
        rubric_provider_debug_logger: RubricProviderDebugLogger | None = None,
        config: InterpreterEnvConfig | None = None,
        input_data: list[dict[str, str | int | None]] | None = None,
        use_host_env_vars: bool = False,
        extra_envs: dict[str, str] | None = None,
        include_env_state_msg: bool = False,
        save_dir: Path | None = None,
    ):
        self.config = config or InterpreterEnvConfig()
        self.work_dir = work_dir
        self.rubric_model = rubric_model
        self.rubric_dispatcher = rubric_dispatcher
        self.rubric_provider_debug_logger = rubric_provider_debug_logger
        self.done = False
        self.problem = problem
        self.use_host_env_vars = use_host_env_vars
        self.extra_envs = extra_envs or {}
        self.save_dir = save_dir

        # Execution config for timeouts and capabilities
        self.execution_config = self.config.execution_config
        self.execution_timeout = self.execution_config.cell_execution_timeout
        self.max_steps = self.config.max_steps

        self.input_data = input_data
        self.output_data: list[dict[str, str | int]] = []
        self.logger = logger
        self.start_time: float | None = None
        self.step_count = 0
        self.include_env_state_msg = include_env_state_msg
        self.state: InterpreterEnvState
        self.kernel_seed: int | None = None
        self.scheduler_seed: int | None = None
        self._kernel_execution_seconds = 0.0
        self._simulated_generation_seconds = 0.0
        self._reported_generation_seconds = 0.0
        self._policy_generation_count = 0
        self._policy_generation_output_tokens = 0
        self._duplicate_generation_accounting_suppressed = 0
        self._accounted_generation_ids: set[str] = set()
        self._unreported_execution_time_count = 0
        self._unreported_execution_observed_seconds = 0.0
        self._duplicate_execution_accounting_suppressed = 0
        self._accounted_execution_ids: set[str] = set()
        self._kernel_timeout_count = 0
        self._kernel_interrupt_success_count = 0
        self._kernel_interrupt_failure_count = 0
        self._kernel_interrupt_seconds_total = 0.0
        self._kernel_interrupt_seconds_max = 0.0
        self._kernel_wedged = False
        # prompting_config is set during reset() after language resolution
        self.prompting_config: PromptingConfig

        if self.score_info_path.exists():
            self.score_info_path.unlink()

        nb_path = self.work_dir / "notebook.ipynb"
        if nb_path.exists():
            nb_path.unlink()

    @property
    def language(self) -> NBLanguage:
        return self.config.language

    async def close(self) -> None:
        """Save notebook, shut down interpreter/container."""
        self.logger.info("Closing environment")
        await self.state.close()

    def _capsule_ref(self) -> CapsuleRef | None:
        """Build the primary backend's capsule delivery reference for this task."""
        capsule_uuid = self.problem.input_data_path or str(self.problem.id)
        spec = self.config.opensandbox_spec
        if spec is not None and spec.capsule_mode == "large_bundle":
            return CapsuleRef(
                uuid=capsule_uuid,
                delivery="bundled",
                image=spec.resolve_large_bundle_image(capsule_uuid),
            )
        if spec is not None and spec.capsule_mode == "mounted_volume" and self.config.pull_capsule_in_pod:
            return CapsuleRef(
                source=spec.mounted_capsule_root,
                uuid=capsule_uuid,
                delivery="mounted_volume",
            )
        if spec is not None and self.config.pull_capsule_in_pod:
            return CapsuleRef(
                source=spec.capsule_source,
                uuid=capsule_uuid,
                delivery="object_store",
            )
        if self.config.k8s_sandbox_specs and self.config.pull_capsule_in_pod:
            return CapsuleRef(uuid=capsule_uuid, delivery="object_store")
        return None

    async def reset(self) -> tuple[Messages, list[Tool]]:
        """Reset the environment and prepare for execution."""
        reset_id = getattr(self, "_nemo_env_id", "?")[:8]
        logger.warning("[reset:%s] building state for work_dir=%s", reset_id, self.work_dir)

        # Format environment capabilities with job_timeout
        env_capabilities = self.execution_config.environment_capabilities_prompt.format(
            job_timeout=self.execution_config.job_timeout
        )

        self.prompting_config = self.config.prompting_config.interpolate(
            language=self.language.value.capitalize(),
            environment_capabilities=env_capabilities,
            job_timeout=self.execution_config.job_timeout,
        )

        # Use kernel environment paths for isolated execution
        kernel_env_path = Path(cfg.KERNEL_ENV_PATH)
        kernel_site_packages = kernel_env_path / "lib" / "python3.12" / "site-packages"

        if self.config.deterministic:
            deterministic_seeds = EnvSeeds.derive(self.config.seed, self.config.env_idx)
            self.kernel_seed = deterministic_seeds.kernel
            self.scheduler_seed = deterministic_seeds.scheduler
        else:
            self.kernel_seed = None
            self.scheduler_seed = None

        self.state = InterpreterEnvState(
            work_dir=self.work_dir,
            language=self.language,
            execution_timeout=self.execution_timeout,
            timeout_recovery=self.execution_config.cell_timeout_recovery,
            interrupt_grace_seconds=self.execution_config.cell_interrupt_grace_seconds,
            safe_execute=self.execution_config.safe_execute,
            use_host_env_vars=self.use_host_env_vars,
            extra_envs={
                # Point to kernel environment's site-packages
                "PYTHONPATH": str(kernel_site_packages),
                # Include kernel environment bin in PATH
                "PATH": (str(kernel_env_path / "bin") + os.pathsep + os.environ.get("PATH", "")),
                # R library path for user-installed packages
                "R_LIBS_USER": str(kernel_env_path / "lib" / "R" / "library"),
            }
            | self.extra_envs,
            save_dir=self.save_dir,
            use_docker=self.config.use_docker,
            use_enroot=self.config.use_enroot,
            use_ray=self.config.use_ray,
            container_sqsh_path=self.config.container_sqsh_path,
            sandbox_memory_request_mb=self.execution_config.sandbox_memory_request_mb,
            sandbox_memory_limit_mb=self.execution_config.sandbox_memory_limit_mb,
            sandbox_max_pids=self.execution_config.sandbox_max_pids,
            sandbox_cpu_request=self.execution_config.sandbox_cpu_request,
            sandbox_cpu=self.execution_config.sandbox_cpu,
            sandbox_ephemeral_storage_gib=self.execution_config.sandbox_ephemeral_storage_gib,
            sandbox_gpu_count=self.execution_config.sandbox_gpu_count,
            sandbox_gpu_type=self.execution_config.sandbox_gpu_type,
            enable_recovery=self.config.enable_recovery,
            k8s_specs=self.config.k8s_sandbox_specs or None,
            opensandbox_spec=self.config.opensandbox_spec,
            sandbox_job_id=self.config.sandbox_job_id,
            # Mirror the dataset's capsule resolution: prefer input_data_path,
            # else problem id. Gated so pure-exec/no-data smokes can disable it.
            capsule_ref=self._capsule_ref(),
            seed=self.kernel_seed,
            scheduler_seed=self.scheduler_seed,
        )
        logger.warning("[reset:%s] starting container", reset_id)
        await self.state.start()
        logger.warning("[reset:%s] container started, building tools", reset_id)

        # Begin episode-time accounting after backend startup, preserving the
        # historical wall-clock boundary while resetting all active-time totals.
        self._reset_episode_time_accounting()

        messages = []
        if self.prompting_config.system_prompt:
            messages.append(Message(role="system", content=self.prompting_config.system_prompt))

        self._filesystem_tool = FilesystemTool(self.work_dir)

        # Reproducibility: wager_mode='off' runs expose the IDENTICAL submit_answer
        # schema they saw before the wager patch landed. Only wager_mode ∈ {shadow,
        # active} builds the closure that adds a `confidence` field. The closure is
        # renamed to "submit_answer" so the tool-call name the policy sees is stable
        # across modes.
        if self.config.wager_mode == "off":
            submit_tool = Tool.from_function(self.submit_answer)
        else:
            base_submit = self.submit_answer  # bound method; captured in closure

            async def _submit_answer_with_wager(answer: str, confidence: float = 0.0) -> str:
                """Submit your response to the research question.

                Note that this tool may only be called once and ends the episode.

                Args:
                    answer: Your final response to the research question.
                    confidence: A wager value in [0.0, 1.0] reflecting how strongly
                        your work supports the answer. 0.0 = fully hedged (you
                        submit an answer but aren't willing to wager on it; full
                        credit if correct, no extra cost if wrong). Larger values
                        stake more on the answer: larger bonus if correct, larger
                        reduction of procedural credit if wrong. 1.0 = maximum
                        wager. Choose the value that reflects how strongly your
                        work supports the answer. Wagering high on answers you
                        cannot defend will cost more than it gains; wagering low
                        on answers you can defend leaves value on the table. If
                        you are unsure, the safe default is a low confidence.
                """
                self.state.wager = clamp_confidence(confidence)
                return await base_submit(answer)

            _submit_answer_with_wager.__name__ = "submit_answer"
            submit_tool = Tool.from_function(_submit_answer_with_wager)

        # Same reproducibility principle for run_cell: when the cell-timeout
        # override is off, the exposed tool is the plain `self.run_cell` whose
        # schema is identical to pre-patch. When on, the closure adds a
        # `timeout_seconds` kwarg clamped to [cell_timeout_min, cell_timeout_max]
        # (default [60, 1200]) and delegates to `_run_cell_with_cap` with the
        # clamped cap.
        if self.config.cell_timeout_override_mode == "off":
            run_cell_tool = Tool.from_function(self.run_cell)
        else:
            ct_min = float(self.config.cell_timeout_min)
            ct_max = float(self.config.cell_timeout_max)
            env_default_cap = float(self.execution_timeout)
            run_cell_impl = self._run_cell_with_cap  # bound method captured in closure

            async def _run_cell_with_timeout(
                code: str,
                idx: int | None = None,
                timeout_seconds: float | None = None,
            ) -> Message | str | list[dict[str, Any]]:
                """Run code in a notebook cell and return the execution output.

                This method allows running code in a new cell (append) or re-running
                an existing cell with updated code.

                Usage Examples:
                    run_cell("print('Hello, world!')")
                    run_cell("print('Hello, world!')", idx=0)
                    run_cell("slow_op()", timeout_seconds=900)

                Error Recovery:
                    When a cell fails with an error, you MUST fix it by calling
                    run_cell with the corrected code and the SAME idx as the failed
                    cell:

                    run_cell("corrected_code", idx=3)  # Fix error in Cell #3

                    The cell number is shown in the output prefix (e.g., "[Cell #3]").
                    Do NOT create a new cell to fix an error - always edit the
                    failed cell.

                Args:
                    code: Code to execute.
                    idx: Cell index to run. If None or >= len(cells), appends a new
                        cell. If provided, updates and re-runs the existing cell at
                        that index. Use this to fix errors in existing cells.
                    timeout_seconds: Optional per-cell execution cap, in seconds.
                        Use this if you expect a long-running cell (e.g., a large
                        DE analysis, a permutation test) to exceed the default cap.
                        Values below the minimum (60s) or above the maximum (1200s)
                        are silently clamped. Leave unset for most cells to use the
                        default cap. A cell that hits its cap returns a TimeoutError
                        output just like any other timeout.

                Returns:
                    Message with multimodal content if images present, otherwise
                    string. The response includes the cell number (e.g.,
                    "[Cell #0] output...").

                Related tools:
                    `reset_kernel` and `list_dir` are separate tools, NOT Python
                    symbols in the kernel namespace. Do NOT write `reset_kernel()`
                    or `list_dir()` inside a `run_cell` call — invoke them as
                    separate tool calls instead. A `TimeoutError` reports whether
                    the cell was interrupted and the kernel is ready.

                Installing packages:
                    Package-manager commands run inside the current sandbox. Check
                    whether a package is already importable before installing it,
                    prefer pip for Python packages, and run `apt-get update` before
                    the first `apt-get install`. Workspace-scoped installs persist
                    across cells and `reset_kernel`, but not across a new sandbox.
                """
                if timeout_seconds is None:
                    cap = env_default_cap
                else:
                    try:
                        cap = float(timeout_seconds)
                    except (TypeError, ValueError):
                        cap = env_default_cap
                    cap = max(ct_min, min(ct_max, cap))
                    self.state.cell_timeout_override_requests.append(cap)
                return await run_cell_impl(code, idx=idx, timeout_cap=cap)

            _run_cell_with_timeout.__name__ = "run_cell"
            run_cell_tool = Tool.from_function(_run_cell_with_timeout)

        self.tools = [
            run_cell_tool,
            Tool.from_function(self.reset_kernel),
            submit_tool,
            Tool.from_function(self.list_dir),
        ]

        messages.append(
            Message(
                content=HYPOTHESIS_TASK_DESC.format(
                    language=self.language.value.capitalize(),
                    hypothesis=self.problem.hypothesis,
                    protocol=self.problem.protocol,
                )
            )
        )

        if self.include_env_state_msg:
            messages.append(self.get_env_state_msg())

        # Always show initial directory listing (with truncation protection)
        messages.append(Message(content=await self.list_dir()))

        logger.warning("[reset:%s] reset fully complete", reset_id)
        return messages, self.tools

    async def step(self, action: ToolRequestMessage) -> tuple[Messages, float, bool, bool]:
        """Execute a step in the environment."""
        # One or more model generations may precede an environment step. In
        # kernel-execution mode, charge them before a tool computes its dynamic
        # execution timeout.
        self._record_policy_generation(action)
        self.step_count += 1
        obs = cast(
            Messages,
            await self.exec_tool_calls(action, concurrency=False, handle_tool_exc=True),
        )

        obs = [*obs]
        if self.include_env_state_msg:
            obs.append(self.get_env_state_msg())

        time_msg = self.get_time_management_message()
        if time_msg is not None:
            obs.append(time_msg)

        if self.step_count >= (self.max_steps - 1):
            obs.append(Message(content=cfg.FORCE_MSG))

        self.state.actions.append(str(action))
        reward = self.state.score if self.state.done else 0.0
        return obs, reward, self.state.done, False

    # ========== Tools ==========

    async def list_dir(
        self,
        directory: str = ".",
        max_files: int = 20,
        show_hidden: bool = False,
    ) -> str:
        """List contents of a directory with truncation protection.

        Recursively lists files in a directory, with built-in protection against
        overwhelming the context with too many files. Use this tool instead of
        writing code to list directories to avoid context bloat.

        Usage Examples:
            list_dir()                      # List working directory
            list_dir("data/")               # List specific folder
            list_dir(max_files=50)          # Show more files
            list_dir(show_hidden=True)      # Include hidden files

        Args:
            directory: Directory path to list (default: current working directory)
            max_files: Maximum number of files to display (default: 20)
            show_hidden: Whether to show hidden files starting with '.' (default: False)
        """
        return await self.state.sandbox.list_dir(directory, max_files, show_hidden)

    async def run_cell(
        self,
        code: str,
        idx: int | None = None,
    ) -> Message | str | list[dict[str, Any]]:
        """Run code in a notebook cell and return the execution output.

        This method allows running code in a new cell (append) or re-running
        an existing cell with updated code.

        Usage Examples:
            run_cell("print('Hello, world!')")           # Run code in new cell
            run_cell("print('Hello, world!')", idx=0)    # Run code in existing cell at index 0

        Error Recovery:
            When a cell fails with an error, you MUST fix it by calling run_cell
            with the corrected code and the SAME idx as the failed cell:

            run_cell("corrected_code", idx=3)  # Fix error in Cell #3

            The cell number is shown in the output prefix (e.g., "[Cell #3]").
            Do NOT create a new cell to fix an error - always edit the failed cell.

        Args:
            code: Code to execute
            idx: Cell index to run. If None or >= len(cells), appends a new cell.
                If provided, updates and re-runs the existing cell at that index.
                Use this to fix errors in existing cells.

        Returns:
            Message with multimodal content if images present, otherwise string.
            The response includes the cell number (e.g., "[Cell #0] output...").

        Related tools:
            `reset_kernel` and `list_dir` are separate tools, NOT Python symbols
            in the kernel namespace. Do NOT write `reset_kernel()` or `list_dir()`
            inside a `run_cell` call — invoke them as separate tool calls instead.
            A `TimeoutError` reports whether the cell was interrupted and the
            kernel is ready.

        Installing packages:
            Package-manager commands run inside the current sandbox. Check whether
            a package is already importable before installing it, prefer pip for
            Python packages, and run `apt-get update` before the first
            `apt-get install`. Workspace-scoped installs persist across cells and
            `reset_kernel`, but not across a new sandbox.
        """
        return await self._run_cell_with_cap(code, idx=idx, timeout_cap=self.execution_timeout)

    async def _run_cell_with_cap(
        self,
        code: str,
        idx: int | None = None,
        timeout_cap: float | None = None,
    ) -> Message | str | list[dict[str, Any]]:
        """Implementation shared by `run_cell` and the timeout-override closure.

        `timeout_cap` caps the per-cell execution time. `run_cell` passes
        `self.execution_timeout` (the config default). The override closure
        passes the model's clamped requested value.
        """
        if timeout_cap is None:
            timeout_cap = self.execution_timeout

        run_cell_uuid = str(uuid.uuid4())
        remaining_seconds = self.get_remaining_time()

        if remaining_seconds <= self.execution_config.force_submit_threshold:
            self.logger.warning(
                f"Refusing cell execution with {remaining_seconds:.1f}s remaining "
                f"(force threshold: {self.execution_config.force_submit_threshold}s)"
            )
            return cfg.FORCE_MSG

        dynamic_timeout = remaining_seconds - self.execution_config.force_submit_threshold
        effective_timeout = min(timeout_cap, dynamic_timeout)

        self.logger.info(
            f"Cell execution with dynamic timeout: {effective_timeout:.1f}s "
            f"(remaining: {remaining_seconds:.1f}s, cap: {timeout_cap}s)"
        )

        # Parse idx (handle string input from LLM)
        cell_idx: int | None = None
        if idx is not None:
            try:
                cell_idx = int(idx)
            except (ValueError, TypeError):
                cell_idx = None

        # Execute code and update notebook atomically
        result, actual_cell_idx = await self._execute_and_account_cell(
            code,
            cell_idx=cell_idx,
            timeout=effective_timeout,
            req_uuid=run_cell_uuid,
        )

        # Build response with cell number
        cell_info = f"[Cell #{actual_cell_idx}] "

        image_count = result.count_images()
        if image_count:
            if self.config.replace_image_payloads_with_placeholders:
                image_word = "image output" if image_count == 1 else "image outputs"
                return (
                    cell_info
                    + result.get_truncated_text()
                    + f"\n[{image_count} {image_word} omitted from policy context]"
                )

            # Format images as data URLs for Message. Aviary validates the image
            # via PIL on construction; a figure with >178M pixels trips
            # PIL.Image.DecompressionBombError. Reshape that to a cell-level
            # error matching the `[Cell #N] Error: ...` shape the model sees for
            # every other kernel error, so the framework's generic
            # "Encountered exception during tool call:" wrapper doesn't fire.
            try:
                images = result.get_images()
                if not images:
                    image_word = "image output" if image_count == 1 else "image outputs"
                    return (
                        cell_info
                        + result.get_truncated_text()
                        + f"\n[{image_count} {image_word} could not be encoded]"
                    )
                image_urls = [f"data:{mime_type};base64,{base64_data}" for mime_type, base64_data in images]
                return Message.create_message(
                    role="tool",
                    text=cell_info + result.get_truncated_text(),
                    images=cast(list[np.ndarray | str], image_urls),
                )
            except Exception as e:
                if type(e).__name__ != "DecompressionBombError" and "DecompressionBombError" not in str(e):
                    raise
                self.logger.warning(
                    "DecompressionBombError on image output for cell %d: %s",
                    actual_cell_idx,
                    e,
                )
                hint = "Hint: reduce the figure size or dpi on plt.savefig / fig.savefig."
                # Replace the cell's image output with an error output so the
                # notebook state is consistent with what the model sees in text.
                if 0 <= actual_cell_idx < len(self.state.nb.cells):
                    self.state.nb.cells[actual_cell_idx].outputs = [
                        nbformat.v4.new_output(
                            output_type="error",
                            ename="DecompressionBombError",
                            evalue=str(e),
                            traceback=[f"DecompressionBombError: {e}", hint],
                        )
                    ]
                return f"{cell_info}Error: DecompressionBombError\nMessage: {e}\nTraceback: {hint}"

        return cell_info + result.get_truncated_text()

    async def reset_kernel(self) -> str:
        """Reset the kernel to a clean state.

        This clears all variables and execution state.
        """
        await self.state.sandbox.reset()

        # Reset notebook state to match kernel reset
        self.state.nb = nbformat.v4.new_notebook()
        self.state.nb.metadata.kernelspec = self.state.language.make_kernelspec()
        self.state.notebook_runtime_errors = []
        self.state._execution_count = 0

        return "Kernel reset successfully."

    @property
    def score_info_path(self) -> Path:
        return self.work_dir / "score_info.json"

    def _record_rubric_model_success(self, response_text: str) -> None:
        self.state.rubric_model_raw_response = response_text
        self.state.rubric_model_failed = False
        self.state.rubric_model_fail_type = ""
        self.state.rubric_model_error_type = ""

    def _record_rubric_model_failure(
        self,
        fail_type: str,
        *,
        response_text: str | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        if response_text is not None:
            self.state.rubric_model_raw_response = response_text
        self.state.rubric_model_failed = True
        self.state.rubric_model_fail_type = fail_type
        if isinstance(error, BaseException):
            self.state.rubric_model_error_type = type(error).__name__
        else:
            self.state.rubric_model_error_type = "" if error is None else str(error)

    @staticmethod
    def _parse_rubric_score(response_text: str) -> int:
        return int(response_text.split("<score>")[1].split("</score>")[0])

    def get_result_metadata(self) -> dict[str, JsonValue]:
        fail_type = self.state.rubric_model_fail_type if self.state.rubric_model_failed else ""
        raw_response = self.state.rubric_model_raw_response or str(self.state.score_metadata.get("response", "") or "")
        score = float(self.state.score)
        zero_reward = score == 0.0
        parsed_score: int | None = None
        if raw_response:
            with contextlib.suppress(Exception):
                parsed_score = self._parse_rubric_score(raw_response)
        rubric_points_awarded = parsed_score if parsed_score is not None else self.state.raw_score
        metadata: dict[str, JsonValue] = {
            "rubric_model_raw_response": raw_response,
            "rubric_model_parsed_score": parsed_score,
            "rubric_model_failed": self.state.rubric_model_failed,
            "rubric_model_fail_type": fail_type,
            "rubric_model_error_type": self.state.rubric_model_error_type,
            "rubric_model_fail_request_error": fail_type == "request_error",
            "rubric_model_fail_parse_error": fail_type == "parse_error",
            "rubric_model_fail_empty_response": fail_type == "empty_response",
            "raw_score": self.state.raw_score,
            "score": score,
            "is_pass_rollout": float(score == 1.0),
            "is_pass90_rollout": float(score >= 0.9),
            "rubric_reward_raw": float(self.state.rubric_reward_raw),
            "zero_reward": zero_reward,
            "positive_rubric_score_zero_reward": rubric_points_awarded > 0 and zero_reward,
            "kernel_timeout_count": self._kernel_timeout_count,
            "kernel_interrupt_success_count": self._kernel_interrupt_success_count,
            "kernel_interrupt_failure_count": self._kernel_interrupt_failure_count,
            "kernel_interrupt_seconds_total": self._kernel_interrupt_seconds_total,
            "kernel_interrupt_seconds_max": self._kernel_interrupt_seconds_max,
            "kernel_wedged": self._kernel_wedged,
            "time_accounting": self.get_time_accounting_metadata(),
            "rubric_dispatch_enabled": self.rubric_dispatcher is not None,
        }
        dispatch_metadata = self.state.score_metadata.get("rubric_dispatch")
        if isinstance(dispatch_metadata, Mapping):
            metadata.update(
                {
                    f"rubric_dispatch_{key}": cast(JsonValue, value)
                    for key, value in dispatch_metadata.items()
                }
            )
        if self.config.deterministic:
            metadata |= {
                "deterministic": True,
                "env_idx": self.config.env_idx,
                "kernel_seed": self.kernel_seed,
                "scheduler_seed": self.scheduler_seed,
                "rubric_seed": self.config.rubric_seed,
            }
        return metadata

    @staticmethod
    def _rubric_image_metadata(rubric_images: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Return image metadata safe for score_info.json; never include data URLs."""
        metadata: list[dict[str, Any]] = []
        for image in rubric_images:
            item = {k: v for k, v in image.items() if k != "data_url"}
            data_url = str(image.get("data_url", ""))
            item["data_url_chars"] = len(data_url)
            metadata.append(item)
        return metadata

    async def _call_rubric_model(
        self,
        prompt: str,
        rubric_images: list[Mapping[str, Any]],
        *,
        timeout: float,
    ) -> Any:
        assert self.rubric_model is not None
        request_kwargs: dict[str, Any] = {"timeout": timeout}
        if self.config.rubric_seed is not None:
            request_kwargs["seed"] = self.config.rubric_seed

        debug_request = None
        if self.rubric_provider_debug_logger is not None:
            debug_request = self.rubric_provider_debug_logger.begin_request(
                prompt=prompt,
                rubric_images=rubric_images,
                env_idx=self.config.env_idx,
                problem_id=str(self.problem.id),
            )
            request_kwargs["metadata"] = self.rubric_provider_debug_logger.metadata_for(debug_request)

        try:
            if not rubric_images:
                response = await self.rubric_model.call_single(prompt, **request_kwargs)
            else:
                image_urls = [str(image["data_url"]) for image in rubric_images]
                message = Message.create_message(
                    role="user",
                    text=prompt,
                    images=cast(list[np.ndarray | str], image_urls),
                )
                response = await self.rubric_model.call_single([message], **request_kwargs)
        except BaseException as error:
            if debug_request is not None:
                self.rubric_provider_debug_logger.finish_request(debug_request, error)
            raise

        if debug_request is not None:
            self.rubric_provider_debug_logger.finish_request(debug_request)
        return response

    async def _evaluate_rubric_once(
        self,
        solution: str,
        nb_content: str,
        rubric_images: list[Mapping[str, Any]],
        *,
        timeout: float,
    ) -> int:
        assert self.rubric_model is not None

        prompt = self.state.score_metadata["prompt"] = RUBRIC_SCORE_PROMPT.format(
            hypothesis=self.problem.hypothesis,
            accepted=self.problem.accepted,
            rubric=self.problem.rubric,
            notebook=nb_content,
            proposed_solution=solution,
        )

        self.state.score_metadata["rubric_images"] = self._rubric_image_metadata(rubric_images)
        try:
            resp = await self._call_rubric_model(prompt, rubric_images, timeout=timeout)
        except Exception as e:
            self._record_rubric_model_failure("request_error", error=e)
            raise

        response_text = str(getattr(resp, "text", "") or "")
        self.state.rubric_model_raw_response = response_text
        if not response_text:
            self._record_rubric_model_failure("empty_response", response_text=response_text)
            raise ValueError("No response from rubric model")
        self.state.score_metadata["response"] = response_text

        try:
            score = self._parse_rubric_score(response_text)
        except Exception as e:
            self._record_rubric_model_failure("parse_error", response_text=response_text, error=e)
            raise ValueError("Failed to parse score from response") from e
        self._record_rubric_model_success(response_text)
        return score

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type(ValueError))
    async def _evaluate_rubric_direct(
        self,
        solution: str,
        nb_content: str,
        rubric_images: list[Mapping[str, Any]],
    ) -> int:
        return await self._evaluate_rubric_once(solution, nb_content, rubric_images, timeout=10 * 60)

    async def _evaluate_rubric(
        self,
        solution: str,
        nb_content: str,
        rubric_images: list[Mapping[str, Any]],
    ) -> int:
        """Evaluate the solution against the rubric. Returns raw integer score."""
        if self.rubric_dispatcher is None:
            return await self._evaluate_rubric_direct(solution, nb_content, rubric_images)

        timeout = self.rubric_dispatcher.config.attempt_timeout_seconds
        try:
            result = await self.rubric_dispatcher.run(
                lambda _attempt: self._evaluate_rubric_once(
                    solution,
                    nb_content,
                    rubric_images,
                    timeout=timeout,
                )
            )
        except RubricDispatchError as error:
            self.state.score_metadata["rubric_dispatch"] = error.metrics.as_dict()
            if not self.state.rubric_model_failed:
                self._record_rubric_model_failure("request_error", error=error)
            raise

        self.state.score_metadata["rubric_dispatch"] = result.metrics.as_dict()
        return result.value

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type(ValueError))
    async def _evaluate_faithfulness_gate(
        self,
        solution: str,
        nb_content: str,
        rubric_images: list[Mapping[str, Any]],
    ) -> bool:
        """Evaluate whether the conclusion is supported by notebook state. Returns True if faithful."""
        assert self.rubric_model is not None

        additional = ""
        if self.problem.faithfulness_rubric:
            additional = f"Additional task-specific criteria:\n{self.problem.faithfulness_rubric}"

        prompt = FAITHFULNESS_GATE_PROMPT.format(
            hypothesis=self.problem.hypothesis,
            notebook=nb_content,
            proposed_solution=solution,
            additional_criteria=additional,
        )
        self.state.faithfulness_metadata["prompt"] = prompt
        self.state.faithfulness_metadata["rubric_images"] = self._rubric_image_metadata(rubric_images)

        resp = await self._call_rubric_model(prompt, rubric_images, timeout=10 * 60)
        if not resp.text:
            raise ValueError("No response from faithfulness gate")
        self.state.faithfulness_metadata["response"] = resp.text

        if "<verdict>PASS</verdict>" in resp.text:
            return True
        if "<verdict>FAIL</verdict>" in resp.text:
            return False
        raise ValueError("Failed to parse verdict from faithfulness gate response")

    async def _evaluate_hybrid_gate(
        self,
        solution: str,
        nb_content: str,
        rubric_images: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Per-item hybrid faithfulness judge. Fail-open on any error.

        The judge reads the raw rubric text and is responsible for numbering
        items and echoing each item's weight inline. We do NOT parse the
        rubric on the client — formats in the dataset are too varied for
        regex to handle reliably.

        Returns a dict with parse_hybrid_response fields plus:
            item_weights, prompt, response, judge_call_failed, parse_failed,
            weights_mismatch (True if sum(weights) != problem.max_score).
        """
        assert self.rubric_model is not None

        prompt = HYBRID_GATE_PROMPT.format(
            hypothesis=self.problem.hypothesis,
            notebook=nb_content,
            proposed_solution=solution,
            rubric=self.problem.rubric,
        )

        try:
            resp = await self._call_rubric_model(prompt, rubric_images, timeout=10 * 60)
        except Exception as e:
            self.logger.exception("Hybrid judge call failed — failing open")
            return {
                "per_item": [],
                "proc_present_pts": 0,
                "proc_max_pts": 0,
                "concl_present_pts": 0,
                "concl_max_pts": 0,
                "item_weights": [],
                "prompt": prompt,
                "rubric_images": self._rubric_image_metadata(rubric_images),
                "response": "",
                "judge_call_failed": True,
                "parse_failed": False,
                "weights_mismatch": False,
                "error": repr(e),
            }

        if not resp.text:
            self.logger.warning("Hybrid judge returned empty response — failing open")
            return {
                "per_item": [],
                "proc_present_pts": 0,
                "proc_max_pts": 0,
                "concl_present_pts": 0,
                "concl_max_pts": 0,
                "item_weights": [],
                "prompt": prompt,
                "rubric_images": self._rubric_image_metadata(rubric_images),
                "response": "",
                "judge_call_failed": True,
                "parse_failed": False,
                "weights_mismatch": False,
                "error": "empty response",
            }

        parsed = parse_hybrid_response(resp.text)

        # Build the 1-indexed weight list the rubric-award synthesis needs.
        # Items may come out of order; fill gaps with 0 (treated as "no such item").
        item_weights: list[int] = []
        if parsed["per_item"]:
            max_idx = max(idx for idx, _, _, _ in parsed["per_item"])
            by_idx = {idx: w for idx, w, _, _ in parsed["per_item"]}
            item_weights = [by_idx.get(i, 0) for i in range(1, max_idx + 1)]

        total_weight = sum(item_weights)
        weights_mismatch = bool(item_weights) and total_weight != self.problem.max_score
        if weights_mismatch:
            self.logger.warning(
                "Hybrid judge weights sum to %d but problem.max_score=%d — failing open on scoring",
                total_weight,
                self.problem.max_score,
            )

        return {
            **parsed,
            "item_weights": item_weights,
            "prompt": prompt,
            "rubric_images": self._rubric_image_metadata(rubric_images),
            "response": resp.text,
            "judge_call_failed": False,
            "parse_failed": not parsed["per_item"],
            "weights_mismatch": weights_mismatch,
        }

    async def _score_solution(self, solution: str) -> bool:
        assert self.rubric_model is not None
        rubric_notebook_serialization = self.config.rubric_notebook_serialization
        if rubric_notebook_serialization == "auto":
            rubric_notebook_serialization = "multimodal" if self.config.include_images_in_rubric_model else "legacy"

        if rubric_notebook_serialization == "legacy":
            nb_content, _ = view_notebook(self.state.nb.cells, self.language.value)
            rubric_images: list[Mapping[str, Any]] = []
        else:
            nb_content, rendered_images = render_notebook_for_rubric(
                self.state.nb.cells,
                self.language.value,
                include_images=self.config.include_images_in_rubric_model,
                max_images=self.config.max_rubric_images,
            )
            # NotebookRubricImage is structurally a Mapping; the cast bridges
            # list invariance for the shared rubric-model helpers.
            rubric_images = cast(list[Mapping[str, Any]], rendered_images)

        mode = self.config.faithfulness_mode
        faith_result: dict[str, Any] | None = None

        if mode == "binary":
            rubric_task = asyncio.ensure_future(self._evaluate_rubric(solution, nb_content, rubric_images))
            gate_task = asyncio.ensure_future(
                self._evaluate_faithfulness_gate(solution, nb_content, rubric_images)
            )
            try:
                raw_score = await rubric_task
            except Exception:
                gate_task.cancel()
                raise
            try:
                self.state.faithfulness_passed = await gate_task
            except Exception:
                self.logger.exception("Binary faithfulness gate failed — falling back to rubric-only scoring")
                self.state.faithfulness_passed = None

        elif mode in {"shadow", "hybrid"}:
            rubric_task = asyncio.ensure_future(self._evaluate_rubric(solution, nb_content, rubric_images))
            hybrid_task = asyncio.ensure_future(self._evaluate_hybrid_gate(solution, nb_content, rubric_images))
            try:
                raw_score = await rubric_task
            except Exception:
                hybrid_task.cancel()
                raise
            try:
                faith_result = await hybrid_task
            except Exception:
                self.logger.exception("Hybrid gate failed — failing open to rubric-only scoring")
                faith_result = {
                    "per_item": [],
                    "item_weights": [],
                    "judge_call_failed": True,
                    "parse_failed": False,
                    "weights_mismatch": False,
                }

        else:  # "off"
            raw_score = await self._evaluate_rubric(solution, nb_content, rubric_images)

        try:
            self.state.raw_score = raw_score
            correct = raw_score == self.problem.max_score
            rubric_score = raw_score / self.problem.max_score if self.config.normalize_reward else raw_score
            rubric_score = max(
                0.0,
                min(1.0 if self.config.normalize_reward else self.problem.max_score, rubric_score),
            )
            self.state.rubric_reward_raw = float(rubric_score)

            if mode == "binary":
                if self.state.faithfulness_passed is False:
                    self.logger.info("Binary faithfulness gate FAILED — zeroing reward")
                    applied = 0.0
                else:
                    applied = rubric_score

            elif mode in {"shadow", "hybrid"}:
                assert faith_result is not None
                item_weights = faith_result.get("item_weights", [])
                judge_broken = (
                    faith_result.get("judge_call_failed", False)
                    or faith_result.get("parse_failed", False)
                    or faith_result.get("weights_mismatch", False)
                )
                if judge_broken or not item_weights:
                    # Fail-open: hybrid reward equals rubric reward, no items stripped.
                    self.state.hybrid_reward_value = float(rubric_score)
                    self.state.hybrid_metadata = {**faith_result, "strip_reason": "judge_unavailable"}
                else:
                    rubric_awards = synthesize_per_item_awards(raw_score, item_weights)
                    hybrid_value, breakdown = hybrid_reward(rubric_awards, faith_result, self.problem.max_score)
                    self.state.hybrid_reward_value = float(hybrid_value)
                    self.state.hybrid_metadata = {**faith_result, **breakdown}
                applied = rubric_score if mode == "shadow" else self.state.hybrid_reward_value

            else:  # "off"
                applied = rubric_score

            # Scheme D: wager-shaped reward. Runs AFTER faithfulness-mode scoring
            # and consumes the gate's correct signal via hybrid_metadata. Shadow
            # mode computes but does not apply. Active mode applies and relaxes
            # the upper clamp so the upside bonus can lift reward above 1.0.
            wager_mode = self.config.wager_mode
            if wager_mode != "off":
                hm = self.state.hybrid_metadata or {}
                proc_max = int(hm.get("proc_max_pts", 0))
                concl_max_hm = int(hm.get("concl_max_pts", 0))
                proc_credited = float(hm.get("proc_pts_credited", 0.0))
                concl_credited = float(hm.get("concl_pts_credited", 0.0))
                gate_unavailable = (proc_max + concl_max_hm) <= 0 or hm.get("strip_reason") == "judge_unavailable"
                if gate_unavailable:
                    self.state.wager_reward_shadow = float(applied)
                    self.state.wager_metadata = {
                        "skipped_reason": "gate_unavailable",
                        "wager": self.state.wager,
                    }
                else:
                    gate_correct = concl_credited >= concl_max_hm > 0
                    wager_value, wager_breakdown = score_with_wager(
                        proc_credit=proc_credited,
                        proc_max=proc_max,
                        concl_credit=concl_credited,
                        concl_max=concl_max_hm,
                        correct=gate_correct,
                        wager=self.state.wager,
                        beta=self.config.wager_beta,
                        gamma=self.config.wager_gamma,
                    )
                    self.state.wager_reward_shadow = float(wager_value)
                    self.state.wager_metadata = wager_breakdown

                if wager_mode == "active":
                    applied = self.state.wager_reward_shadow

            # Upper clamp relaxes only when wager is active — the bonus can
            # legitimately lift reward above 1.0, and downstream (NeMo-RL
            # advantage computation) does not re-clamp.
            if wager_mode == "active":
                applied = max(0.0, applied)
            # In off/shadow the existing clamp is preserved implicitly (the
            # computed `applied` already came out of rubric_score/hybrid paths
            # which were clamped above).

            self.state.score = applied
            self.state.total_reward += applied
            self.state.zero_reward = float(self.state.score) == 0.0
            return correct

        finally:
            self.state.zero_reward = float(self.state.score) == 0.0
            score_info = {
                **self.state.score_metadata,
                "score": self.state.score,
                "raw_score": self.state.raw_score,
                "max_score": self.problem.max_score,
                "rubric_model_raw_response": self.state.rubric_model_raw_response,
                "rubric_model_failed": self.state.rubric_model_failed,
                "rubric_model_fail_type": self.state.rubric_model_fail_type,
                "rubric_model_error_type": self.state.rubric_model_error_type,
                "zero_reward": self.state.zero_reward,
                "faithfulness_passed": self.state.faithfulness_passed,
                "faithfulness_mode": self.config.faithfulness_mode,
                "rubric_reward_raw": self.state.rubric_reward_raw,
                "hybrid_reward_value": self.state.hybrid_reward_value,
                "hybrid_metadata": self.state.hybrid_metadata,
                "wager_mode": self.config.wager_mode,
                "wager": self.state.wager,
                "wager_reward_shadow": self.state.wager_reward_shadow,
                "wager_metadata": self.state.wager_metadata,
                "time_accounting": self.get_time_accounting_metadata(),
            }
            if self.config.deterministic:
                score_info |= {
                    "deterministic": True,
                    "env_idx": self.config.env_idx,
                    "base_seed": self.config.seed,
                    "kernel_seed": self.kernel_seed,
                    "scheduler_seed": self.scheduler_seed,
                    "rubric_seed": self.config.rubric_seed,
                }
            with self.score_info_path.open("w") as f:
                json.dump(score_info, f, indent=2, default=str)

            self.logger.info(f"Received solution ({self.state.raw_score}/{self.problem.max_score}): {solution!r}.")

    async def submit_answer(self, answer: str) -> str:
        """Submit your response to the research question.

        Note that this tool may only be called once and ends the episode.

        Args:
            answer: Your final response to the research question
        """
        if self.state.done or self.state.answer is not None:
            return "Episode already finished."

        self.state.answer = answer

        if self.rubric_model is None:
            self.logger.warning("No rubric_model configured, skipping scoring")
            self.state.done = True
            return answer

        try:
            correct = await self._score_solution(answer)
        finally:
            self.state.done = True
        return CORRECT_MSG if correct else INCORRECT_MSG

    # ========== Time Management ==========

    def _reset_episode_time_accounting(self) -> None:
        """Reset wall, kernel-execution, and model-generation counters."""
        self.start_time = time.perf_counter()
        self._kernel_execution_seconds = 0.0
        self._simulated_generation_seconds = 0.0
        self._reported_generation_seconds = 0.0
        self._policy_generation_count = 0
        self._policy_generation_output_tokens = 0
        self._duplicate_generation_accounting_suppressed = 0
        self._accounted_generation_ids.clear()
        self._unreported_execution_time_count = 0
        self._unreported_execution_observed_seconds = 0.0
        self._duplicate_execution_accounting_suppressed = 0
        self._accounted_execution_ids.clear()
        self._kernel_timeout_count = 0
        self._kernel_interrupt_success_count = 0
        self._kernel_interrupt_failure_count = 0
        self._kernel_interrupt_seconds_total = 0.0
        self._kernel_interrupt_seconds_max = 0.0
        self._kernel_wedged = False

    def _record_policy_generation(self, action: ToolRequestMessage | None = None) -> float:
        """Charge model turns preceding an environment step and return their seconds."""
        accounting = self.execution_config.time_accounting
        latency = (
            accounting.generation_latency
            if isinstance(accounting, cfg.KernelExecutionTimeAccountingConfig)
            else None
        )
        turns = model_turns_from_action_info(action.info if action is not None else None)

        _validate_generation_measurements(latency, turns)

        # Legacy callers do not provide turn metadata. Preserve one generation
        # per environment step for all pre-existing accounting modes.
        candidates: list[ModelTurn | None] = [*turns] if turns else [None]
        unique_turns: list[ModelTurn | None] = []
        seen_in_request: set[str] = set()
        for turn in candidates:
            response_id = None if turn is None else turn.response_id
            if response_id is not None and (
                response_id in self._accounted_generation_ids or response_id in seen_in_request
            ):
                self._duplicate_generation_accounting_suppressed += 1
                self.logger.warning("Suppressed duplicate generation-time charge for response %s", response_id)
                continue
            if response_id is not None:
                seen_in_request.add(response_id)
            unique_turns.append(turn)

        self._accounted_generation_ids.update(seen_in_request)
        self._policy_generation_count += len(unique_turns)
        output_tokens = sum(
            turn.usage.output_tokens
            for turn in unique_turns
            if turn is not None and turn.usage is not None
        )
        self._policy_generation_output_tokens += output_tokens

        seconds = _generation_seconds(latency, unique_turns, output_tokens)
        if isinstance(latency, cfg.ReportedGenerationLatencyConfig):
            self._reported_generation_seconds += seconds
        else:
            self._simulated_generation_seconds += seconds
        return seconds

    def _record_kernel_execution(
        self,
        reported_seconds: float | None,
        observed_seconds: float,
        *,
        logical_execution_id: str,
    ) -> float:
        """Record one logical cell once, never charging client-side wait/retry time."""
        if logical_execution_id in self._accounted_execution_ids:
            self._duplicate_execution_accounting_suppressed += 1
            self.logger.warning("Suppressed duplicate execution-time charge for request %s", logical_execution_id)
            return 0.0
        self._accounted_execution_ids.add(logical_execution_id)

        if reported_seconds is None or not math.isfinite(reported_seconds) or reported_seconds < 0:
            observed = observed_seconds if math.isfinite(observed_seconds) and observed_seconds > 0 else 0.0
            self._unreported_execution_time_count += 1
            self._unreported_execution_observed_seconds += observed
            self.logger.warning(
                "Execution backend returned invalid execution_time=%r; not charging %.3fs of client-observed wait time",
                reported_seconds,
                observed,
            )
            return 0.0

        seconds = float(reported_seconds)
        self._kernel_execution_seconds += seconds
        return seconds

    def _record_timeout_recovery(self, result: ExecutionResult) -> None:
        if not result.timed_out:
            return
        self._kernel_timeout_count += 1
        if result.timeout_recovery == "interrupted":
            self._kernel_interrupt_success_count += 1
        elif result.timeout_recovery == "wedged":
            self._kernel_interrupt_failure_count += 1
            self._kernel_wedged = True
        if result.interrupt_seconds is not None:
            self._kernel_interrupt_seconds_total += result.interrupt_seconds
            self._kernel_interrupt_seconds_max = max(
                self._kernel_interrupt_seconds_max,
                result.interrupt_seconds,
            )

    async def _execute_and_account_cell(
        self,
        code: str,
        *,
        cell_idx: int | None,
        timeout: float | None,
        req_uuid: str = "",
    ) -> tuple[ExecutionResult, int]:
        """Execute one cell and add its kernel-reported duration to the episode clock."""
        logical_execution_id = req_uuid or str(uuid.uuid4())
        observed_start = time.perf_counter()
        result, actual_cell_idx = await self.state.execute_and_add_cell(
            code,
            cell_idx=cell_idx,
            timeout=timeout,
            req_uuid=logical_execution_id,
        )
        self._record_kernel_execution(
            result.execution_time,
            time.perf_counter() - observed_start,
            logical_execution_id=logical_execution_id,
        )
        self._record_timeout_recovery(result)
        return result, actual_cell_idx

    def get_elapsed_time(self) -> float:
        """Return elapsed episode time under the configured accounting policy."""
        accounting = self.execution_config.time_accounting
        if isinstance(accounting, cfg.WallClockTimeAccountingConfig):
            return 0.0 if self.start_time is None else max(0.0, time.perf_counter() - self.start_time)
        return self._kernel_execution_seconds + self._simulated_generation_seconds + self._reported_generation_seconds

    def get_time_accounting_metadata(self) -> dict[str, Any]:
        """Return auditable wall and accounted-time totals for rollout artifacts."""
        accounting = self.execution_config.time_accounting
        wall_elapsed = 0.0 if self.start_time is None else max(0.0, time.perf_counter() - self.start_time)
        elapsed = (
            wall_elapsed
            if isinstance(accounting, cfg.WallClockTimeAccountingConfig)
            else self._kernel_execution_seconds
            + self._simulated_generation_seconds
            + self._reported_generation_seconds
        )
        generation_latency = (
            accounting.generation_latency.model_dump(mode="json")
            if isinstance(accounting, cfg.KernelExecutionTimeAccountingConfig)
            else None
        )
        return {
            "mode": accounting.mode,
            "budget_seconds": self.execution_config.job_timeout,
            "elapsed_seconds": elapsed,
            "remaining_seconds": self.execution_config.job_timeout - elapsed,
            "wall_clock_elapsed_seconds": wall_elapsed,
            "kernel_execution_seconds": self._kernel_execution_seconds,
            "model_generation_seconds": self._simulated_generation_seconds + self._reported_generation_seconds,
            "simulated_generation_seconds": self._simulated_generation_seconds,
            "reported_generation_seconds": self._reported_generation_seconds,
            "policy_generation_count": self._policy_generation_count,
            "policy_generation_output_tokens": self._policy_generation_output_tokens,
            "duplicate_generation_accounting_suppressed": self._duplicate_generation_accounting_suppressed,
            "unreported_execution_time_count": self._unreported_execution_time_count,
            "unreported_execution_observed_seconds": self._unreported_execution_observed_seconds,
            "duplicate_execution_accounting_suppressed": self._duplicate_execution_accounting_suppressed,
            "generation_latency": generation_latency,
        }

    def get_remaining_time(self) -> int:
        """Get remaining execution time in seconds."""
        return int(self.execution_config.job_timeout - self.get_elapsed_time())

    def get_time_management_message(self) -> Message | None:
        """Get a time management message if thresholds are reached."""
        remaining = self.get_remaining_time()

        if remaining <= self.execution_config.force_submit_threshold:
            self.logger.warning(
                f"Forcing answer submission with {remaining}s remaining "
                f"(threshold: {self.execution_config.force_submit_threshold}s)"
            )
            return Message(content=cfg.FORCE_MSG.format(remaining=remaining))

        if remaining <= self.execution_config.warn_submit_threshold:
            self.logger.info(
                f"Warning agent about timeout with {remaining}s remaining "
                f"(threshold: {self.execution_config.warn_submit_threshold}s)"
            )
            return Message(content=cfg.WARN_MSG.format(remaining=remaining))

        return None

    # ========== State Export ==========

    def export_frame(self) -> Frame:
        """Export the current environment state as a Frame."""
        return Frame(
            state={
                "last_action": self.state.actions[-1] if self.state.actions else None,
                "answer": self.state.answer,
                "done": self.state.done,
                "nb_state": self.state.nb,
                "nb_runtime_errors": self.state.notebook_runtime_errors,
                "raw_score": self.state.raw_score,
                "score": self.state.score,
                "score_metadata": self.state.score_metadata,
                "rubric_model_raw_response": self.state.rubric_model_raw_response,
                "rubric_model_failed": self.state.rubric_model_failed,
                "rubric_model_fail_type": self.state.rubric_model_fail_type,
                "rubric_model_error_type": self.state.rubric_model_error_type,
                "zero_reward": self.state.zero_reward,
                "total_reward": self.state.total_reward,
                "faithfulness_passed": self.state.faithfulness_passed,
                "faithfulness_metadata": self.state.faithfulness_metadata,
                "rubric_reward_raw": self.state.rubric_reward_raw,
                "hybrid_reward_value": self.state.hybrid_reward_value,
                "hybrid_metadata": self.state.hybrid_metadata,
                "faithfulness_mode": self.config.faithfulness_mode,
                "wager": self.state.wager,
                "wager_reward_shadow": self.state.wager_reward_shadow,
                "wager_metadata": self.state.wager_metadata,
                "wager_mode": self.config.wager_mode,
            },
            info={
                "language": self.state.language,
                "problem": self.problem,
                "work_dir": self.work_dir,
                "input_data": self.input_data,
                "output_data": self.output_data,
                "time_accounting": self.get_time_accounting_metadata(),
            },
        )

    def get_env_state_msg(self) -> EnvStateMessage:
        """Get the current environment state message."""
        summary = self.state.get_execution_summary()

        state_summary = (
            f"{summary['language']} Interpreter Environment\n"
            f"Working Directory: {summary['work_dir']}\n"
            f"Execution History: {summary['total_executions']} commands executed\n"
        )

        if summary["recent_errors"]:
            state_summary += "\nRecent Errors:\n"
            for error in summary["recent_errors"]:
                state_summary += f"- {error}\n"

        if summary["last_execution"]:
            max_len = 200
            state_summary += "\nLast Execution:\n"
            last_exec = summary["last_execution"]
            # Get code from the last notebook cell (ExecutionResult doesn't store code)
            if self.state.nb.cells:
                last_cell = self.state.nb.cells[-1]
                code_source = last_cell.get("source", "")
                code = code_source[:max_len] + "..." if len(code_source) > max_len else code_source
                state_summary += f"Code: {code}\n"
            # Use ExecutionResult methods
            text_outputs = last_exec.get_text_outputs()
            if text_outputs:
                output = text_outputs[0]
                output = output[:max_len] + "..." if len(output) > max_len else output
                state_summary += f"Output: {output}\n"
            if last_exec.has_images():
                images_count = len(last_exec.get_images())
                state_summary += f"Images generated: {images_count}\n"

        return EnvStateMessage.create_message(text=state_summary, images=[])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir")
    args = parser.parse_args()

    work_dir = Path(args.work_dir or mkdtemp())
    print(f"Working directory: {work_dir}")

    problem = ProblemInstance(
        id="",
        hypothesis="",
        protocol="",
        answer=False,
        rubric="",
        max_points=0,
        metadata={},
    )

    env = InterpreterEnv(problem=problem, work_dir=work_dir, config=InterpreterEnvConfig(use_docker=True))
    await env.reset()

    code = ""
    done = False
    while not done:
        breakpoint()  # noqa: T100
        action = ToolRequestMessage(tool_calls=[ToolCall.from_name("run_cell", code=code)])
        obs, *_ = await env.step(action)
        for msg in obs:
            print(msg.content)

    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
