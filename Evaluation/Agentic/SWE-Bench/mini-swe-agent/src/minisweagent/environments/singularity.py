#!/usr/bin/env python3

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class SingularityEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Environment variables to forward to the container."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_SINGULARITY_EXECUTABLE", "singularity")
    """Path to the singularity executable."""
    sandbox_build_retries: int = 3
    """Number of retries for building the sandbox if an error occurs."""
    global_args: list[str] = ["--quiet"]
    """Global arguments passed before the subcommand (e.g., --quiet, --debug)."""
    exec_args: list[str] = ["--contain", "--cleanenv", "--fakeroot"]
    """Arguments passed to `singularity exec`."""
    overlay_size_mb: int = 10240
    """Size of the ext3 overlay image in MB when using a local sandbox directory.
    All writes go to the overlay, keeping the original sandbox clean."""


class SingularityEnvironment:
    def __init__(
        self, *, config_class: type = SingularityEnvironmentConfig, logger: logging.Logger | None = None, **kwargs
    ):
        """Singularity environment. See `SingularityEnvironmentConfig` for kwargs."""
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self._is_local_sandbox = Path(self.config.image).is_dir()
        self._overlay_dir: Path | None = None
        self.sandbox_dir = self._build_sandbox()

    def _build_sandbox(self) -> Path:
        # If the image is already a local sandbox directory, use it directly
        # with an ext3 overlay so the original stays clean
        image_path = Path(self.config.image)
        if image_path.is_dir():
            self.logger.info(f"Using local sandbox with overlay: {image_path}")
            self._overlay_dir = Path(tempfile.mkdtemp(prefix="minisweagent_overlay_"))
            overlay_path = self._overlay_dir / "overlay.img"
            self.logger.info(f"Creating {self.config.overlay_size_mb}MB ext3 overlay at {overlay_path}")
            subprocess.run(
                ["dd", "if=/dev/zero", f"of={overlay_path}", "bs=1M", f"count={self.config.overlay_size_mb}"],
                check=True, capture_output=True,
            )
            subprocess.run(
                ["mkfs.ext3", "-q", "-F", str(overlay_path)],
                check=True, capture_output=True,
            )
            return image_path

        # Building the sandbox can fail (very rarely), so we retry it
        max_retries = self.config.sandbox_build_retries
        for attempt in range(max_retries):
            sandbox_dir = Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"
            try:
                subprocess.run(
                    [self.config.executable, "build", "--sandbox", sandbox_dir, self.config.image],
                    check=True,
                    capture_output=True,
                )
                break
            except subprocess.CalledProcessError as e:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                self.logger.error(
                    f"Error building image {self.config.image}, stdout: {e.stdout}, stderr: {e.stderr} (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    raise
        return sandbox_dir

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in a Singularity container and return the result as a dict."""
        command = action.get("command", "")
        cmd = [self.config.executable, *self.config.global_args, "exec", *self.config.exec_args]

        work_dir = cwd or self.config.cwd
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        # Use overlay for local sandboxes (read-only sandbox + writable overlay),
        # otherwise use --writable on the built sandbox copy
        if self._overlay_dir is not None:
            overlay_path = self._overlay_dir / "overlay.img"
            cmd.extend(["--overlay", str(overlay_path), str(self.sandbox_dir), "bash", "-c", command])
        else:
            cmd.extend(["--writable", str(self.sandbox_dir), "bash", "-c", command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "MINI_SWE_AGENT_FINAL_OUTPUT") and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        # Clean up overlay image if used
        if self._overlay_dir is not None:
            shutil.rmtree(self._overlay_dir, ignore_errors=True)
            self._overlay_dir = None
        else:
            # Only remove sandbox dir if we built it (not a local sandbox)
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def __del__(self):
        """Cleanup sandbox when object is destroyed."""
        self.cleanup()
