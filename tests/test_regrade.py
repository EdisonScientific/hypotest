"""Tests for trajectory reconstruction in hypotest.regrade."""

from uuid import uuid4

from aviary.core import Message, ToolCall, ToolRequestMessage
from ldp.agent.simple_agent import SimpleAgentState
from ldp.data_structures import Trajectory, Transition
from ldp.graph import OpResult
from ldp.graph.ops import CallID

from hypotest.regrade import reconstruct_notebook


def _make_op(action: ToolRequestMessage) -> OpResult:
    call_id = CallID(run_id=uuid4(), fwd_id=uuid4())
    return OpResult(call_id=call_id, op_name="test", op_class_name="test", value=action)


def _make_step(
    tool_calls: list[ToolCall],
    obs_contents: list[str],
    timestep: int = 0,
) -> Transition:
    action = ToolRequestMessage(tool_calls=tool_calls)
    obs = [Message(content=c) for c in obs_contents]
    state = SimpleAgentState(messages=[], tools=[])
    return Transition(
        timestep=timestep,
        agent_state=state,
        next_agent_state=state,
        action=_make_op(action),
        observation=[],
        next_observation=obs,
    )


class TestReconstructNotebook:
    def test_single_cell(self):
        step = _make_step(
            [ToolCall.from_name("run_cell", code="print(1)")],
            ["[Cell #0] [stdout] 1"],
        )
        traj = Trajectory(steps=[step], traj_id="task_0")
        notebook, answer = reconstruct_notebook(traj)

        assert "print(1)" in notebook
        assert "[Cell #0] [stdout] 1" in notebook
        assert answer is None

    def test_multiple_cells_sequential(self):
        steps = [
            _make_step(
                [ToolCall.from_name("run_cell", code="import pandas as pd")],
                ["[Cell #0] Code executed successfully (no output)"],
                timestep=0,
            ),
            _make_step(
                [ToolCall.from_name("run_cell", code="df = pd.read_csv('data.csv')")],
                ["[Cell #1] Code executed successfully (no output)"],
                timestep=1,
            ),
            _make_step(
                [ToolCall.from_name("run_cell", code="print(df.shape)")],
                ["[Cell #2] [stdout] (100, 5)"],
                timestep=2,
            ),
        ]
        traj = Trajectory(steps=steps, traj_id="task_0")
        notebook, answer = reconstruct_notebook(traj)

        assert "### Cell 0:" in notebook
        assert "### Cell 1:" in notebook
        assert "### Cell 2:" in notebook
        assert "import pandas as pd" in notebook
        assert "df = pd.read_csv('data.csv')" in notebook
        assert "print(df.shape)" in notebook
        assert "(100, 5)" in notebook
        assert answer is None

    def test_cell_overwrite(self):
        """When idx is provided, the cell at that index should be overwritten."""
        steps = [
            _make_step(
                [ToolCall.from_name("run_cell", code="prnt(1)")],
                ["[Cell #0] Error: NameError"],
                timestep=0,
            ),
            _make_step(
                [ToolCall.from_name("run_cell", code="print(1)", idx=0)],
                ["[Cell #0] [stdout] 1"],
                timestep=1,
            ),
        ]
        traj = Trajectory(steps=steps, traj_id="task_0")
        notebook, _answer = reconstruct_notebook(traj)

        assert "prnt(1)" not in notebook
        assert "print(1)" in notebook
        assert "[stdout] 1" in notebook
        assert "NameError" not in notebook

    def test_submit_answer(self):
        steps = [
            _make_step(
                [ToolCall.from_name("run_cell", code="result = 42")],
                ["[Cell #0] Code executed successfully (no output)"],
                timestep=0,
            ),
            _make_step(
                [ToolCall.from_name("submit_answer", answer="The answer is 42")],
                ["Correct answer!"],
                timestep=1,
            ),
        ]
        traj = Trajectory(steps=steps, traj_id="task_0")
        notebook, answer = reconstruct_notebook(traj)

        assert "result = 42" in notebook
        assert answer == "The answer is 42"

    def test_parallel_tool_calls(self):
        """Multiple run_cell calls in a single step should be handled correctly."""
        step = _make_step(
            [
                ToolCall.from_name("run_cell", code="import numpy as np"),
                ToolCall.from_name("run_cell", code="import pandas as pd"),
                ToolCall.from_name("run_cell", code="print('hello')"),
            ],
            [
                "[Cell #0] Code executed successfully (no output)",
                "[Cell #1] Code executed successfully (no output)",
                "[Cell #2] [stdout] hello",
            ],
        )
        traj = Trajectory(steps=[step], traj_id="task_0")
        notebook, _answer = reconstruct_notebook(traj)

        assert "### Cell 0:" in notebook
        assert "### Cell 1:" in notebook
        assert "### Cell 2:" in notebook
        assert "import numpy" in notebook
        assert "import pandas" in notebook
        assert "hello" in notebook

    def test_non_run_cell_tools_ignored(self):
        """Tools like list_dir should not affect notebook reconstruction."""
        steps = [
            _make_step(
                [ToolCall.from_name("list_dir", path=".")],
                ["Files in directory: data.csv, README.md"],
                timestep=0,
            ),
            _make_step(
                [ToolCall.from_name("run_cell", code="print('ok')")],
                ["[Cell #0] [stdout] ok"],
                timestep=1,
            ),
        ]
        traj = Trajectory(steps=steps, traj_id="task_0")
        notebook, _answer = reconstruct_notebook(traj)

        assert "list_dir" not in notebook
        assert "### Cell 0:" in notebook
        assert "print('ok')" in notebook

    def test_mixed_parallel_with_submit(self):
        """run_cell and submit_answer in the same step."""
        step = _make_step(
            [
                ToolCall.from_name("run_cell", code="x = 1 + 1"),
                ToolCall.from_name("submit_answer", answer="x is 2"),
            ],
            [
                "[Cell #0] Code executed successfully (no output)",
                "Correct answer!",
            ],
        )
        traj = Trajectory(steps=[step], traj_id="task_0")
        notebook, answer = reconstruct_notebook(traj)

        assert "x = 1 + 1" in notebook
        assert answer == "x is 2"

    def test_cell_overwrite_preserves_later_cells(self):
        """Overwriting cell 1 should not affect cell 2."""
        steps = [
            _make_step(
                [ToolCall.from_name("run_cell", code="a = 1")],
                ["[Cell #0] Code executed successfully (no output)"],
                timestep=0,
            ),
            _make_step(
                [ToolCall.from_name("run_cell", code="b = 'original'")],
                ["[Cell #1] Code executed successfully (no output)"],
                timestep=1,
            ),
            _make_step(
                [ToolCall.from_name("run_cell", code="c = 3")],
                ["[Cell #2] Code executed successfully (no output)"],
                timestep=2,
            ),
            _make_step(
                [ToolCall.from_name("run_cell", code="b = 20", idx=1)],
                ["[Cell #1] Code executed successfully (no output)"],
                timestep=3,
            ),
        ]
        traj = Trajectory(steps=steps, traj_id="task_0")
        notebook, _answer = reconstruct_notebook(traj)

        assert "a = 1" in notebook
        assert "b = 'original'" not in notebook
        assert "b = 20" in notebook
        assert "c = 3" in notebook

    def test_empty_trajectory(self):
        traj = Trajectory(steps=[], traj_id="task_0")
        notebook, answer = reconstruct_notebook(traj)

        assert not notebook
        assert answer is None

    def test_no_output_cell_has_no_output_block(self):
        """Cells with empty output should not produce an Output section."""
        step = _make_step(
            [ToolCall.from_name("run_cell", code="x = 1")],
            [""],
        )
        traj = Trajectory(steps=[step], traj_id="task_0")
        notebook, _answer = reconstruct_notebook(traj)

        assert "### Cell 0:" in notebook
        assert "x = 1" in notebook
        assert "### Output 0:" not in notebook

    def test_idx_string_coercion(self):
        """Idx passed as a string (from LLM) should be coerced to int."""
        steps = [
            _make_step(
                [ToolCall.from_name("run_cell", code="old")],
                ["[Cell #0] out"],
                timestep=0,
            ),
            _make_step(
                [ToolCall.from_name("run_cell", code="new", idx="0")],
                ["[Cell #0] new out"],
                timestep=1,
            ),
        ]
        traj = Trajectory(steps=steps, traj_id="task_0")
        notebook, _answer = reconstruct_notebook(traj)

        assert "old" not in notebook
        assert "new" in notebook
