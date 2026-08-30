"""PRoot-backed execution for SIF images on containerized schedulers.

Beaker jobs run inside an OCI container without the namespace capabilities
needed by nested Apptainer.  Apptainer can still pull and unpack a SIF there,
so this environment uses it only for extraction and executes the resulting
writable root filesystem with PRoot.  The agent therefore sees the same image
contents as the published Singularity recipe without requiring a Docker daemon
or privileged nested container runtime.
"""

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


class ProotSIFEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    env: dict[str, str] = {}
    forward_env: list[str] = []
    timeout: int = 30
    executable: str = os.getenv("MSWEA_PROOT_EXECUTABLE", "proot")
    apptainer_executable: str = os.getenv("MSWEA_APPTAINER_EXECUTABLE", "apptainer")
    unsquashfs_executable: str = os.getenv("MSWEA_UNSQUASHFS_EXECUTABLE", "unsquashfs")
    sandbox_build_retries: int = 3


class ProotSIFEnvironment:
    def __init__(
        self,
        *,
        config_class: type[ProotSIFEnvironmentConfig] = ProotSIFEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.sandbox_dir = self._build_sandbox()

    def _build_sandbox(self) -> Path:
        for attempt in range(self.config.sandbox_build_retries):
            sandbox_dir = Path(tempfile.gettempdir()) / f"minisweagent-proot-{uuid.uuid4().hex[:8]}"
            squashfs_path = sandbox_dir.with_suffix(".squashfs")
            try:
                sif_listing = subprocess.run(
                    [self.config.apptainer_executable, "sif", "list", self.config.image],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                filesystem_ids = [
                    line.split()[0]
                    for line in sif_listing.stdout.splitlines()
                    if "FS" in line and line.split() and line.split()[0].isdigit()
                ]
                if not filesystem_ids:
                    raise RuntimeError(f"No filesystem partition found in SIF: {self.config.image}")
                with squashfs_path.open("wb") as squashfs_stream:
                    subprocess.run(
                        [
                            self.config.apptainer_executable,
                            "sif",
                            "dump",
                            filesystem_ids[-1],
                            self.config.image,
                        ],
                        check=True,
                        stdout=squashfs_stream,
                        stderr=subprocess.PIPE,
                    )
                subprocess.run(
                    [
                        self.config.unsquashfs_executable,
                        "-d",
                        str(sandbox_dir),
                        str(squashfs_path),
                    ],
                    check=True,
                    capture_output=True,
                )
                squashfs_path.unlink(missing_ok=True)
                return sandbox_dir
            except (subprocess.CalledProcessError, RuntimeError) as error:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                squashfs_path.unlink(missing_ok=True)
                self.logger.error(
                    "Error extracting image %s from SIF, stdout: %s, stderr: %s "
                    "(attempt %s/%s)",
                    self.config.image,
                    getattr(error, "stdout", ""),
                    getattr(error, "stderr", ""),
                    attempt + 1,
                    self.config.sandbox_build_retries,
                )
                if attempt == self.config.sandbox_build_retries - 1:
                    raise
        raise RuntimeError("unreachable")

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
        command = action.get("command", "")
        work_dir = cwd or self.config.cwd
        clean_env = {
            "HOME": "/root",
            "LANG": "C.UTF-8",
            "PATH": "/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                clean_env[key] = value
        clean_env.update(self.config.env)

        cmd = [
            self.config.executable,
            "-0",
            "-R",
            str(self.sandbox_dir),
            "-w",
            work_dir,
            "/usr/bin/env",
            "-i",
            *(f"{key}={value}" for key, value in clean_env.items()),
            "bash",
            "-c",
            command,
        ]
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
        except Exception as error:
            raw_output = getattr(error, "output", None)
            if isinstance(raw_output, bytes):
                raw_output = raw_output.decode("utf-8", errors="replace")
            output = {
                "output": raw_output or "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {error}",
                "extra": {"exception_type": type(error).__name__, "exception": str(error)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self) -> None:
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def __del__(self):
        self.cleanup()
