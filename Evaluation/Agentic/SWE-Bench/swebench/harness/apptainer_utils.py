from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path

HEREDOC_DELIMITER = "EOF_1399519320"  # different from dataset HEREDOC_DELIMITERs!

# Default directory for storing sandbox images
SANDBOX_CACHE_DIR = Path("logs/sandboxes")
SANDBOX_WORK_DIR = Path("logs/sandbox_workdirs")


def get_sandbox_cache_dir() -> Path:
    """Get the sandbox cache directory, creating it if needed."""
    cache_dir = Path(os.environ.get("SWEBENCH_SANDBOX_DIR", str(SANDBOX_CACHE_DIR)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_sandbox_work_dir() -> Path:
    """Get the sandbox working directory for per-run copies."""
    work_dir = Path(os.environ.get("SWEBENCH_SANDBOX_WORK_DIR", str(SANDBOX_WORK_DIR)))
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def sanitize_image_name(image_name: str) -> str:
    """Convert a Docker-style image name to a valid directory name."""
    return image_name.replace(":", "__").replace("/", "_")


def get_sandbox_path(image_name: str) -> Path:
    """Get the sandbox directory path for a given image name."""
    return get_sandbox_cache_dir() / sanitize_image_name(image_name)


def sandbox_exists(image_name: str) -> bool:
    """Check if a sandbox for the given image name already exists."""
    return get_sandbox_path(image_name).is_dir()


def copy_to_sandbox(sandbox_path: Path, src: Path, dst: Path):
    """
    Copy a file from local filesystem into a sandbox directory.

    Args:
        sandbox_path (Path): Path to the sandbox directory
        src (Path): Source file path on host
        dst (Path): Destination path inside the sandbox (absolute, e.g. /tmp/patch.diff)
    """
    if os.path.dirname(dst) == "":
        raise ValueError(
            f"Destination path parent directory cannot be empty!, dst: {dst}"
        )

    # Convert absolute container path to path within sandbox
    # e.g. /tmp/patch.diff -> sandbox_path/tmp/patch.diff
    dst_str = str(dst)
    if dst_str.startswith("/"):
        dst_str = dst_str[1:]
    sandbox_dst = sandbox_path / dst_str

    # Create parent directory if needed
    sandbox_dst.parent.mkdir(parents=True, exist_ok=True)

    # Copy the file
    shutil.copy2(str(src), str(sandbox_dst))


def write_to_sandbox(sandbox_path: Path, data: str, dst: Path):
    """
    Write a string to a file inside a sandbox directory.

    Args:
        sandbox_path (Path): Path to the sandbox directory
        data (str): String data to write
        dst (Path): Destination path inside the sandbox (absolute)
    """
    dst_str = str(dst)
    if dst_str.startswith("/"):
        dst_str = dst_str[1:]
    sandbox_dst = sandbox_path / dst_str
    sandbox_dst.parent.mkdir(parents=True, exist_ok=True)
    sandbox_dst.write_text(data)


def exec_run_with_timeout(
    sandbox_path: Path,
    cmd: str,
    timeout: int | None = 60,
    workdir: str | None = None,
    fakeroot: bool = True,
    bind_mounts: list[tuple[str, str]] | None = None,
):
    """
    Run a command inside an Apptainer sandbox with a timeout.

    Uses --writable-tmpfs so all writes go to a temporary in-memory overlay.
    The original sandbox is never modified.

    Args:
        sandbox_path (Path): Path to the sandbox directory
        cmd (str): Command to run
        timeout (int): Timeout in seconds
        workdir (str): Working directory inside the container
        fakeroot (bool): Whether to use --fakeroot
        bind_mounts (list): List of (host_path, container_path) tuples to bind-mount
    """
    exec_result = b""
    process = None
    exception = None
    timed_out = False

    apptainer_cmd = [
        "apptainer", "exec",
        "--writable-tmpfs",
        "--contain",
        "--no-mount", "tmp",
        "--no-home",
    ]

    if fakeroot:
        apptainer_cmd.append("--fakeroot")

    if bind_mounts:
        for host_path, container_path in bind_mounts:
            apptainer_cmd.extend(["--bind", f"{host_path}:{container_path}"])

    if workdir:
        apptainer_cmd.extend(["--pwd", workdir])

    apptainer_cmd.append(str(sandbox_path))
    apptainer_cmd.extend(["/bin/bash", "-c", cmd])

    def run_command():
        nonlocal exec_result, process, exception
        try:
            process = subprocess.Popen(
                apptainer_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            exec_result, _ = process.communicate()
        except Exception as e:
            exception = e

    thread = threading.Thread(target=run_command)
    start_time = time.time()
    thread.start()
    thread.join(timeout)

    if exception:
        raise exception

    if thread.is_alive():
        # Kill the process group
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        timed_out = True
        thread.join(5)  # Give it a moment to clean up

    end_time = time.time()
    return exec_result.decode(errors="replace"), timed_out, end_time - start_time


def remove_sandbox(sandbox_path: Path, logger=None):
    """
    Remove a sandbox directory.

    Args:
        sandbox_path (Path): Path to sandbox to remove
        logger: Logger or None
    """
    if not sandbox_path or not sandbox_path.exists():
        return

    if not logger:
        log_info = print
        log_error = print
    elif logger == "quiet":
        log_info = lambda x: None
        log_error = lambda x: None
    else:
        log_error = logger.info
        log_info = logger.info

    try:
        log_info(f"Removing sandbox {sandbox_path}...")
        shutil.rmtree(str(sandbox_path))
        log_info(f"Sandbox {sandbox_path} removed.")
    except Exception as e:
        log_error(f"Failed to remove sandbox {sandbox_path}: {e}\n{traceback.format_exc()}")


def cleanup_sandbox(sandbox_path: Path, logger=None):
    """
    Clean up a sandbox (alias for remove_sandbox for API compatibility).
    """
    remove_sandbox(sandbox_path, logger)


def list_sandboxes(cache_dir: Path = None) -> set:
    """
    List all sandboxes in the cache directory.
    Returns a set of image names (reversing the sanitization: __ -> :, _ -> /).
    """
    if cache_dir is None:
        cache_dir = get_sandbox_cache_dir()
    if not cache_dir.exists():
        return set()

    sandboxes = set()
    for entry in cache_dir.iterdir():
        if entry.is_dir():
            # Convert directory name back to image name
            name = entry.name.replace("__", ":")
            sandboxes.add(name)
    return sandboxes


def should_remove(image_name: str, cache_level: str, clean: bool, prior_images: set):
    """
    Determine if a sandbox should be removed based on cache level and clean flag.
    (Same logic as Docker version)
    """
    existed_before = image_name in prior_images
    if "/" in image_name:
        image_name = image_name.rsplit("/", 1)[-1]
    if image_name.startswith("sweb.base"):
        if cache_level in {"none"} and (clean or not existed_before):
            return True
    elif image_name.startswith("sweb.env"):
        if cache_level in {"none", "base"} and (clean or not existed_before):
            return True
    elif image_name.startswith("sweb.eval"):
        if cache_level in {"none", "base", "env"} and (clean or not existed_before):
            return True
    return False


def clean_sandboxes(
    cache_dir: Path, prior_images: set, cache_level: str, clean: bool
):
    """
    Clean sandboxes based on cache level and clean flag.
    """
    images = list_sandboxes(cache_dir)
    removed = 0
    print("Cleaning cached sandboxes...")
    for image_name in images:
        if should_remove(image_name, cache_level, clean, prior_images):
            try:
                sandbox_path = cache_dir / sanitize_image_name(image_name)
                remove_sandbox(sandbox_path, "quiet")
                removed += 1
            except Exception as e:
                print(f"Error removing sandbox {image_name}: {e}")
                continue
    print(f"Removed {removed} sandboxes.")
