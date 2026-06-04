from typing import TypedDict, Optional
import re
import traceback
import uuid
from pathlib import Path

from typing import Dict, Any
from loguru import logger

from jinja2 import Template

from minisweagent.environments import Environment, get_environment


class MiniSWEEvaluationResult(TypedDict):
    instance_id: str
    resolved: bool
    eval_error: Optional[str]


def get_sb_environment(config: dict, instance: dict, data_source: str) -> Environment:
    import copy
    env_config = copy.deepcopy(config.setdefault("environment", {}))
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_docker_image_name(instance, data_source)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        # Check for a local SIF file first to avoid pulling from Docker Hub.
        # `local_sif_dir` may be a single directory (str) or a list of
        # directories searched in order. Multiple dirs let us mix image
        # families in one run, e.g. swe_gym_images + swe_smith_images.
        local_sif_dir = env_config.get("local_sif_dir", "")
        search_dirs = [local_sif_dir] if isinstance(local_sif_dir, str) else list(local_sif_dir)
        search_dirs = [d for d in search_dirs if d]
        # Convert docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-5885:latest
        # to xingyaoww_sweb.eval.x86_64.getmoto_s_moto-5885-latest.sif
        sif_name = image_name.replace("docker.io/", "").replace("/", "_").replace(":", "-") + ".sif"
        resolved_sif = next((Path(d) / sif_name for d in search_dirs if (Path(d) / sif_name).exists()), None)
        if resolved_sif is not None:
            env_config["image"] = str(resolved_sif)
        else:
            if search_dirs:
                logger.warning(
                    f"Local SIF {sif_name} not found in {search_dirs}, falling back to docker://{image_name}"
                )
            env_config["image"] = f"docker://{image_name}"
    # Remove local_sif_dir before passing to get_environment (not a valid SingularityEnvironmentConfig field)
    env_config.pop("local_sif_dir", None)
    env = get_environment(env_config)
    try:
        if startup_command := config.get("run", {}).get("env_startup_command"):
            startup_command = Template(startup_command).render(**instance)
            out = env.execute({"command": startup_command})
            if out["returncode"] != 0:
                raise RuntimeError(f"Error executing startup command: {out}")
    except Exception:
        env.cleanup()
        raise
    return env


def get_docker_image_name(instance: dict, data_source: str) -> str:
    """Get the image name for a SWEBench/SWE-Gym instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        iid = instance["instance_id"]
        if "swe-gym" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_s_")  # to comply with docker image naming convention
            image_name = f"docker.io/xingyaoww/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        elif "swe-bench" in data_source.lower():
            # Docker doesn't allow double underscore, so we replace them with a magic token
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        else:
            raise NotImplementedError(f"Data source: {data_source} is not supported")
    return image_name


def _strip_pip_installs(eval_script: str) -> str:
    """Replace pip install commands in eval scripts with safe offline alternatives.

    SIF images already have all packages editable-installed. We only need to
    re-run ``pip install -e`` (with --no-build-isolation --no-deps) so that
    code changes from ``git apply`` are picked up. Every other pip/pip3
    invocation is replaced with ``true`` to avoid network access.
    """
    # The editable-install replacement (keeps code changes visible to the test runner)
    _EDITABLE_INSTALL = "pip install -e /testbed --no-build-isolation --no-deps"

    def _process_line(line: str) -> str:
        # Only touch lines that actually contain a pip install invocation
        if "pip install" not in line:
            return line

        # Lines are semicolon-separated compound commands.  Process each
        # segment independently so that non-pip segments (sed, echo, export …)
        # are preserved.
        segments = line.split(";")
        new_segments = []
        has_editable = False
        for seg in segments:
            stripped = seg.strip()
            # Detect pip/pip3/python -m pip install
            if re.search(r"(?:python -m |)pip3?\s+install", stripped):
                if " -e " in stripped or " -e." in stripped:
                    # Editable install — keep one safe version
                    if not has_editable:
                        new_segments.append(f" {_EDITABLE_INSTALL}")
                        has_editable = True
                    # else: drop duplicate editable installs
                else:
                    # Non-editable pip install — skip
                    new_segments.append(" true")
            else:
                new_segments.append(seg)
        return ";".join(new_segments)

    return "\n".join(_process_line(l) for l in eval_script.split("\n"))


def evaluate_trajectory(
    instance: Dict[str, Any], model_patch: str, sweagent_config: dict, data_source: str
) -> MiniSWEEvaluationResult:

    ret = MiniSWEEvaluationResult(instance_id=instance["instance_id"], resolved=False, eval_error=None)

    env = None
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)

        # apply git patch
        # NOTE (sumanthrh): This applies patch in-line, and the maximum patch size is limited by the OS limits for `ARG_MAX`.
        # In modern systems, this is typically ~ 1 MB, which is pretty generous.
        # For simplicity, we assume that large patches greater than `ARG_MAX` are meant to fail
        delimiter = f"PATCH_{uuid.uuid4().hex}"  # unlikely to collide with symbols in the patch
        command = f"git apply <<'{delimiter}'\n{model_patch}\n{delimiter}"

        obs = env.execute({"command": command})

        if obs["returncode"] != 0:
            ret["eval_error"] = obs["output"]
        else:
            # run eval script in-line
            eval_script = instance["eval_script"]
            # Skip pip install commands that trigger PEP 517 build isolation, which
            # downloads setuptools from PyPI and fails in network-isolated Singularity
            # containers. The SIF images already have packages editable-installed with
            # all dependencies, so code changes from git apply are picked up automatically.
            eval_script = eval_script.replace("make init", "true")
            eval_script = _strip_pip_installs(eval_script)
            eval_cmd = f"bash <<'EOF'\n{eval_script}\nEOF"
            # add longer timeout for evaluation
            obs = env.execute({"command": eval_cmd}, timeout=3600)
            # use the return value
            ret["resolved"] = obs["returncode"] == 0
            # truncate to last 1000 characters for brevity
            ret["eval_error"] = (
                f"(truncated to last 1000 characters)\n{obs["output"][-1000:]}" if not ret["resolved"] else None
            )
    except Exception as e:
        ret["eval_error"] = f"Env creation failed with {e}"
        logger.info(f"Starting environment failed with exception: {e}\n, {traceback.format_exc()}")
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception:
                logger.debug(f"Error cleaning up eval environment for {instance['instance_id']}", exc_info=True)

    return ret
