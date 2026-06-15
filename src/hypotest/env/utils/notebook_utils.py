"""Notebook utility functions for heron."""

import asyncio
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypedDict

import nbformat
from aiodocker.containers import DockerContainer

from hypotest.env import config as cfg

from .img_utils import encode_image_to_base64_with_mime

if TYPE_CHECKING:
    from jupyter_client.asynchronous.client import AsyncKernelClient

logger = logging.getLogger(__name__)

JUPYTER_IMAGE_OUTPUT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
}

JUPYTER_TABLE_OUTPUT_TYPES_TO_IGNORE = {
    "text/latex",
    "text/html",
    "text/markdown",
    "application/vnd.jupyter.widget-view+json",
}


class NotebookRubricImage(TypedDict):
    id: str
    cell_idx: int
    output_idx: int
    image_idx: int
    mime_type: str
    data_url: str


def limit_notebook_output(output: str | list[str]) -> str:
    """Limit notebook output to configured length.

    Args:
        output: String output from notebook cell

    Returns:
        String output, truncated if longer than configured limit with
        indication of truncation
    """
    if isinstance(output, list):
        raise TypeError("Only string output truncation is supported")
    output_length = len(output)
    if output_length < cfg.NB_OUTPUT_LIMIT:
        return output
    cutoff = int(cfg.NB_OUTPUT_LIMIT / 2)
    # Sometimes error tracebacks have important information at the end
    # and at the beginning so important to keep those sections
    return (
        output[:cutoff]
        + (
            f"\n<...output truncated to {cfg.NB_OUTPUT_LIMIT} characters"
            " (edit the cell to produce a shorter output if needed)...>\n"
        )
        + output[-cutoff:]
    )


def process_cell_output(
    output: nbformat.NotebookNode,
    md: list[str],
    images: list[str],
    cell_streams: list[str],
    include_images: bool = True,
) -> None:
    """Process a single output from a notebook cell."""
    if output.output_type == "stream":
        cell_streams.append(output.text)
    elif output.output_type == "execute_result":
        data = output.get("data", {}).get("text/plain", "")
        md.append(limit_notebook_output(data))
    elif output.output_type == "error":
        traceback_str = "\n".join(output.traceback) if isinstance(output.traceback, list) else output.traceback
        md.append(limit_notebook_output(traceback_str))
    elif output.output_type in {"display_data"}.union(JUPYTER_IMAGE_OUTPUT_TYPES):
        data_type = next(iter(output.data.keys()), "")
        if data_type in JUPYTER_TABLE_OUTPUT_TYPES_TO_IGNORE:
            return
        if data_type == "text/plain":
            md.append(limit_notebook_output(output.data[data_type]))
        elif data_type in JUPYTER_IMAGE_OUTPUT_TYPES:
            if not include_images:
                md.append("<image output omitted from context>")
                return
            try:
                mime_type, encoded = encode_image_to_base64_with_mime(output.data[data_type], data_type)
                images.append(f"data:{mime_type};base64,{encoded}")
            except RuntimeError:
                logger.exception("Error encoding image.")
                md.append(
                    "\n<The generated image is too large to encode,"
                    " (edit the cell to produce a smaller image if needed)>\n"
                )
            else:
                md.append(f"<{len(images)}>")
        else:
            logger.warning(f"Unknown data type: {data_type}")
            md.append(limit_notebook_output(output.data[data_type]))


def view_notebook(
    cells: list[nbformat.NotebookNode],
    language: str,
    include_images: bool = True,
) -> tuple[str, list[str]]:
    """Process notebook cells and convert them to markdown format with images.

    Args:
        cells: List of notebook cells to process
        language: Programming language of the notebook code cells
        include_images: Whether to return image data URLs in the image list.

    Returns:
        tuple containing:
            - Markdown string with cell contents and outputs
            - List of base64 encoded images found in cell outputs
    """
    md: list[str] = []
    images: list[str] = []

    for idx, cell in enumerate(cells):
        md.append(f"### Cell {idx}:")
        if cell.cell_type == "code":
            md.extend((f"```{language}", str(cell.source), "```"))

            outputs = cell.get("outputs", [])
            if outputs:
                md.extend([f"### Output {idx}:", "```"])
                cell_streams: list[str] = []

                for output in outputs:
                    process_cell_output(output, md, images, cell_streams, include_images=include_images)

                if cell_streams:
                    combined_stream = "\n".join(cell_streams)
                    md.append(limit_notebook_output(combined_stream))
                md.append("```")
        elif cell.cell_type in {"markdown", "raw"}:
            md.append(str(cell.source))

    return "\n".join(md), images


def _format_rubric_image_placeholder(image: NotebookRubricImage) -> str:
    return (
        f'<image id="{image["id"]}" '
        f'cell="{image["cell_idx"]}" '
        f'output="{image["output_idx"]}" '
        f'mime_type="{image["mime_type"]}">'
    )


def _process_rubric_output(
    output: nbformat.NotebookNode,
    *,
    cell_idx: int,
    output_idx: int,
    md: list[str],
    images: list[NotebookRubricImage],
    cell_streams: list[str],
    include_images: bool,
    max_images: int,
) -> None:
    output_type = output.get("output_type", "")
    if output_type == "stream":
        cell_streams.append(output.get("text", ""))
        return

    if output_type == "error":
        traceback = output.get("traceback", [])
        traceback_str = "\n".join(traceback) if isinstance(traceback, list) else traceback
        md.append(limit_notebook_output(traceback_str))
        return

    if output_type not in {"execute_result", "display_data"}:
        return

    data: dict[str, Any] = output.get("data", {})
    text_plain = data.get("text/plain")
    if text_plain:
        md.append(limit_notebook_output(text_plain))

    image_idx = 0
    for data_type in sorted(JUPYTER_IMAGE_OUTPUT_TYPES):
        if data_type not in data:
            continue
        image_idx += 1
        image_id = f"cell-{cell_idx}-output-{output_idx}-image-{image_idx}"
        if not include_images:
            md.append(f'<image id="{image_id}" omitted="context">')
            continue
        if len(images) >= max_images:
            md.append(f'<image id="{image_id}" omitted="max_images_exceeded">')
            continue

        try:
            mime_type, encoded = encode_image_to_base64_with_mime(data[data_type], data_type)
        except RuntimeError:
            logger.exception("Error encoding image for rubric.")
            md.append(f'<image id="{image_id}" omitted="encoding_failed">')
            continue

        image: NotebookRubricImage = {
            "id": image_id,
            "cell_idx": cell_idx,
            "output_idx": output_idx,
            "image_idx": image_idx,
            "mime_type": mime_type,
            "data_url": f"data:{mime_type};base64,{encoded}",
        }
        images.append(image)
        md.append(_format_rubric_image_placeholder(image))

    for data_type, value in data.items():
        if data_type in JUPYTER_IMAGE_OUTPUT_TYPES or data_type == "text/plain":
            continue
        if data_type in JUPYTER_TABLE_OUTPUT_TYPES_TO_IGNORE:
            continue
        if isinstance(value, str):
            md.append(limit_notebook_output(value))


def render_notebook_for_rubric(
    cells: list[nbformat.NotebookNode],
    language: str,
    *,
    include_images: bool = True,
    max_images: int = 20,
) -> tuple[str, list[NotebookRubricImage]]:
    """Render notebook text plus ordered image data for multimodal rubric calls."""
    md: list[str] = []
    images: list[NotebookRubricImage] = []

    for cell_idx, cell in enumerate(cells):
        md.append(f"### Cell {cell_idx}:")
        if cell.cell_type == "code":
            md.extend((f"```{language}", str(cell.source), "```"))

            outputs = cell.get("outputs", [])
            if outputs:
                md.extend([f"### Output {cell_idx}:", "```"])
                cell_streams: list[str] = []

                for output_idx, output in enumerate(outputs):
                    _process_rubric_output(
                        output,
                        cell_idx=cell_idx,
                        output_idx=output_idx,
                        md=md,
                        images=images,
                        cell_streams=cell_streams,
                        include_images=include_images,
                        max_images=max_images,
                    )

                if cell_streams:
                    combined_stream = "\n".join(cell_streams)
                    md.append(limit_notebook_output(combined_stream))
                md.append("```")
        elif cell.cell_type in {"markdown", "raw"}:
            md.append(str(cell.source))

    return "\n".join(md), images


async def nbformat_run_notebook(  # noqa: D417
    cells: Iterable[nbformat.NotebookNode],
    client: "AsyncKernelClient",
    cell_idx: int | None = None,
) -> list[str]:
    """Execute notebook cells using a kernel client and collect outputs.

    Args:
        cells: Notebook cell dictionaries to execute sequentially
        client: KernelClient instance to use for code execution

    Raises:
        ValueError: If there is an error executing a cell

    Returns:
        List of error messages from cells that raised an error
    """
    error_messages = []
    logger.debug(f"Running notebook with cell_idx: {cell_idx}")
    try:
        logger.debug("Beginning cell execution")
        for idx, cell in enumerate(cells):
            if cell_idx is not None and idx != cell_idx:
                logger.debug(f"Skipping cell {idx} because cell_idx is {cell_idx}")
                continue
            if cell.cell_type == "code":
                logger.debug(f"Executing code cell {idx}")
                cell.outputs = []  # Initialize empty outputs list
                msg_id = client.execute(cell.source)
                logger.debug(f"Message ID for cell {idx}: {msg_id}")

                while True:
                    msg = await client.get_iopub_msg()
                    logger.debug(f"Received message type: {msg['msg_type']}")

                    if msg["parent_header"].get("msg_id") == msg_id:
                        msg_type = msg["msg_type"]
                        content = msg["content"]

                        if msg_type in {
                            "execute_result",
                            "display_data",
                            "stream",
                        }:
                            if msg_type == "stream":
                                output = nbformat.v4.new_output(
                                    output_type="stream",
                                    name=content["name"],
                                    text=content["text"],
                                )
                            elif msg_type == "execute_result":
                                output = nbformat.v4.new_output(
                                    output_type="execute_result",
                                    data=content.get("data", {}),
                                    metadata=content.get("metadata", {}),
                                    execution_count=content.get("execution_count"),
                                )
                            else:  # display_data
                                output = nbformat.v4.new_output(
                                    output_type="display_data",
                                    data=content.get("data", {}),
                                    metadata=content.get("metadata", {}),
                                )
                            cell.outputs.append(output)
                            logger.debug(f"Added output of type {msg_type} to cell {idx}")

                        elif msg_type == "error":
                            # Create error output and add it to cell outputs
                            error_output = nbformat.v4.new_output(
                                output_type="error",
                                ename=content.get("ename", ""),
                                evalue=content.get("evalue", ""),
                                traceback=content.get("traceback", []),
                            )
                            cell.outputs.append(error_output)

                            error_msg = (
                                f"Error executing cell {idx}:\n"
                                f"Name: {content.get('ename', 'Unknown')}\n"
                                f"Value: {content.get('evalue', 'No error message')}\n"
                                f"Traceback: {content.get('traceback', [])}"
                            )
                            error_messages.append(f"Cell {idx}: {content.get('evalue', '')}")
                            logger.error(error_msg)
                            # raise ValueError(error_msg)
                        elif msg_type == "status" and content["execution_state"] == "idle":
                            logger.debug(f"Cell {idx} execution finished")
                            break
    finally:
        logger.debug("Stopping kernel channels")
        client.stop_channels()

    return error_messages


async def exec_cmd(
    container: DockerContainer,
    exec_command: list[str],
    timeout: float | None = 300,  # noqa: ASYNC109
) -> tuple[int, str, str]:
    """Execute a command in a Docker container and capture output.

    Args:
        container: Docker container instance to execute command in
        exec_command: Command to execute as list of strings
        timeout: Maximum time in seconds to wait for command completion

    Returns:
        tuple containing:
            - Exit code from command execution
            - stdout output as string
            - stderr output as string

    Raises:
        TimeoutError: If command execution exceeds timeout period
    """
    try:
        async with asyncio.timeout(timeout):
            exec_instance = await container.exec(
                cmd=exec_command,
                tty=True,
                privileged=True,
            )

            # Start the execution
            stream = exec_instance.start()
            stdout = ""
            stderr = ""

            while True:
                try:
                    message = await stream.read_out()
                    if message is None:
                        break

                    # Messages come as tuples of (stream_type, data)
                    stream_type, data = message
                    if stream_type == cfg.DOCKER_STREAM_TYPE_STDOUT:  # stdout
                        stdout += data.decode()
                    elif stream_type == cfg.DOCKER_STREAM_TYPE_STDERR:  # stderr
                        stderr += data.decode()

                except EOFError:
                    break

            exit_code = (await exec_instance.inspect())["ExitCode"]
            logger.debug(f"Command output:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
            return exit_code, stdout, stderr
    except TimeoutError as err:
        raise TimeoutError(f"Command execution timed out after {timeout} seconds") from err


def collect_notebook_stats(  # noqa: PLR0912
    nb: nbformat.NotebookNode,
) -> dict[str, int]:
    """Count lines, cells, outputs, and different language usage in a Jupyter notebook."""
    stats = {
        "code_lines": 0,
        "comment_lines": 0,
        "markdown_lines": 0,
        "code_cells": 0,
        "markdown_cells": 0,
        "images": 0,
        "tables": 0,
        "r_cells": 0,
        "bash_cells": 0,
        "shell_commands": 0,
    }
    for cell in nb.cells:
        # Split cell source into lines and count non-empty lines
        lines = [line for line in cell.source.split("\n") if line.strip()]

        if cell.cell_type == "code":
            stats["code_cells"] += 1

            # Process each line in code cells
            for line in lines:
                line = line.strip()  # noqa: PLW2901
                # Check if line is a comment (starts with # but not #!)
                if line.startswith("#") and not line.startswith("#!"):
                    stats["comment_lines"] += 1
                else:
                    stats["code_lines"] += 1

            # Check for R and bash cells
            if lines:
                first_line = lines[0].strip()
                if first_line.startswith("%%R"):
                    stats["r_cells"] += 1
                elif first_line.startswith("%%bash"):
                    stats["bash_cells"] += 1

                # Count shell commands (lines starting with !)
                stats["shell_commands"] += sum(1 for line in lines if line.strip().startswith("!"))

            # Check outputs for images and tables
            if hasattr(cell, "outputs"):
                for output in cell.outputs:
                    # Check for images
                    if output.get("output_type") in {"display_data", "execute_result"}:
                        if "image/png" in output.get("data", {}):
                            stats["images"] += 1

                        # Check for HTML tables or DataFrame representations
                        if "text/html" in output.get("data", {}):
                            html_content = output["data"]["text/html"]
                            if isinstance(html_content, list):
                                html_content = "".join(html_content)
                            if "<table" in html_content:
                                stats["tables"] += 1

                        # Check for plain text DataFrame representations
                        elif "text/plain" in output.get("data", {}):
                            text_content = output["data"]["text/plain"]
                            if isinstance(text_content, list):
                                text_content = "".join(text_content)
                            if any(marker in text_content for marker in ("DataFrame", "Series")):
                                stats["tables"] += 1

        elif cell.cell_type == "markdown":
            stats["markdown_lines"] += len(lines)
            stats["markdown_cells"] += 1

            # Count markdown images
            for line in lines:
                if "![" in line or "<img" in line:
                    stats["images"] += 1
    return stats
