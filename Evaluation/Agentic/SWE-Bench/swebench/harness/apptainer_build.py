from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import traceback

from pathlib import Path

from swebench.harness.constants import (
    BASE_IMAGE_BUILD_DIR,
    DOCKER_USER,
    ENV_IMAGE_BUILD_DIR,
    INSTANCE_IMAGE_BUILD_DIR,
    UTF8,
)
from swebench.harness.apptainer_utils import (
    cleanup_sandbox,
    get_sandbox_cache_dir,
    remove_sandbox,
    sanitize_image_name,
)
from swebench.harness.test_spec.test_spec import (
    get_test_specs_from_dataset,
    make_test_spec,
    TestSpec,
)
from swebench.harness.utils import ansi_escape, run_threadpool


class BuildImageError(Exception):
    def __init__(self, image_name, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.image_name = image_name
        self.log_path = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {self.super_str}\n"
            f"Check ({self.log_path}) for more information."
        )


def setup_logger(instance_id: str, log_file: Path, mode="w", add_stdout: bool = False):
    """
    This logger is used for logging the build process of images and containers.
    It writes logs to the log file.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{instance_id}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode, encoding=UTF8)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    if add_stdout:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f"%(asctime)s - {instance_id} - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def dockerfile_to_def(dockerfile: str, build_dir: Path, cache_dir: Path = None) -> str:
    """
    Convert a Dockerfile string to an Apptainer definition file string.

    Handles: FROM, RUN, COPY, ENV, ARG, WORKDIR, USER

    Args:
        dockerfile: Dockerfile content string
        build_dir: Build context directory (for COPY sources)
        cache_dir: Directory where sandbox images are cached

    Returns:
        Apptainer definition file content string
    """
    if cache_dir is None:
        cache_dir = get_sandbox_cache_dir()

    lines = dockerfile.strip().split("\n")

    # Parse Dockerfile
    header_lines = []
    post_lines = []
    env_lines = []
    files_lines = []
    current_workdir = None

    # Join continuation lines (backslash at end of line)
    joined_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        while line.rstrip().endswith("\\") and i + 1 < len(lines):
            line = line.rstrip()[:-1] + lines[i + 1]
            i += 1
        joined_lines.append(line)
        i += 1

    for line in joined_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # FROM instruction
        if stripped.upper().startswith("FROM"):
            from_match = re.match(
                r"FROM\s+(?:--platform=\S+\s+)?(\S+)", stripped, re.IGNORECASE
            )
            if from_match:
                image_ref = from_match.group(1)
                # Check if this is a local SWE-bench image (sweb.base.*, sweb.env.*)
                if image_ref.startswith("sweb."):
                    local_sandbox = cache_dir / sanitize_image_name(image_ref)
                    header_lines.append("Bootstrap: localimage")
                    header_lines.append(f"From: {local_sandbox}")
                else:
                    header_lines.append("Bootstrap: docker")
                    header_lines.append(f"From: {image_ref}")
            continue

        # ARG instruction
        if stripped.upper().startswith("ARG "):
            arg_content = stripped[4:].strip()
            if "=" in arg_content:
                var, val = arg_content.split("=", 1)
                post_lines.append(f"export {var}={val}")
            continue

        # ENV instruction
        if stripped.upper().startswith("ENV "):
            env_content = stripped[4:].strip()
            if "=" in env_content:
                # Handle ENV KEY=VALUE
                env_lines.append(f"export {env_content}")
                post_lines.append(f"export {env_content}")
            else:
                # Handle ENV KEY VALUE (space-separated, legacy format)
                parts = env_content.split(None, 1)
                if len(parts) == 2:
                    env_lines.append(f"export {parts[0]}={parts[1]}")
                    post_lines.append(f"export {parts[0]}={parts[1]}")
            continue

        # RUN instruction
        if stripped.upper().startswith("RUN "):
            run_content = stripped[4:].strip()
            if current_workdir:
                post_lines.append(f"cd {current_workdir}")
            post_lines.append(run_content)
            continue

        # COPY instruction
        if stripped.upper().startswith("COPY "):
            copy_content = stripped[5:].strip()
            parts = copy_content.split()
            if len(parts) >= 2:
                src = parts[0]
                dst = parts[1]
                files_lines.append(f"{build_dir}/{src} {dst}")
            continue

        # WORKDIR instruction
        if stripped.upper().startswith("WORKDIR "):
            current_workdir = stripped[8:].strip()
            post_lines.append(f"mkdir -p {current_workdir}")
            post_lines.append(f"cd {current_workdir}")
            continue

        # USER instruction - mostly ignore, fakeroot handles root
        if stripped.upper().startswith("USER "):
            continue

    # Build the definition file
    def_content = []

    # Header
    for hl in header_lines:
        def_content.append(hl)
    def_content.append("")

    # Files section
    if files_lines:
        def_content.append("%files")
        for fl in files_lines:
            def_content.append(f"    {fl}")
        def_content.append("")

    # Post section
    if post_lines:
        def_content.append("%post")
        # Re-export environment variables at the start of %post
        # so they're available during build
        for el in env_lines:
            def_content.append(f"    {el}")
        for pl in post_lines:
            def_content.append(f"    {pl}")
        def_content.append("")

    # Environment section
    if env_lines:
        def_content.append("%environment")
        for el in env_lines:
            def_content.append(f"    {el}")
        def_content.append("")

    return "\n".join(def_content)


def build_image(
    image_name: str,
    setup_scripts: dict,
    dockerfile: str,
    platform: str,
    build_dir: Path,
    nocache: bool = False,
):
    """
    Build an Apptainer sandbox from a Dockerfile.

    Args:
        image_name (str): Name of the image (used for sandbox directory name)
        setup_scripts (dict): Dictionary of setup script names to contents
        dockerfile (str): Contents of the Dockerfile
        platform (str): Platform (unused for Apptainer on native arch, kept for API compat)
        build_dir (Path): Directory for the build context
        nocache (bool): Whether to force rebuild
    """
    cache_dir = get_sandbox_cache_dir()
    sandbox_path = cache_dir / sanitize_image_name(image_name)

    # Create logger
    logger = setup_logger(image_name, build_dir / "build_image.log")
    logger.info(
        f"Building sandbox {image_name}\n"
        f"Using dockerfile:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to build"
    )

    for setup_script_name, setup_script in setup_scripts.items():
        logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")

    try:
        # Write setup scripts to build directory
        build_dir.mkdir(parents=True, exist_ok=True)
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)
            if setup_script_name not in dockerfile:
                logger.warning(
                    f"Setup script {setup_script_name} may not be used in Dockerfile"
                )

        # Convert Dockerfile to Apptainer definition
        def_content = dockerfile_to_def(dockerfile, build_dir, cache_dir)
        def_path = build_dir / "image.def"
        with open(def_path, "w") as f:
            f.write(def_content)

        logger.info(f"Apptainer definition file:\n{def_content}")

        # Remove existing sandbox if nocache
        if nocache and sandbox_path.exists():
            logger.info(f"Removing existing sandbox {sandbox_path} (nocache=True)")
            shutil.rmtree(str(sandbox_path))

        # Build the sandbox
        logger.info(f"Building sandbox at {sandbox_path}")
        build_cmd = [
            "apptainer", "build",
            "--fakeroot",
            "--sandbox",
            str(sandbox_path),
            str(def_path),
        ]

        result = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for builds
        )

        # Log output
        if result.stdout:
            for line in result.stdout.split("\n"):
                if line.strip():
                    logger.info(ansi_escape(line))
        if result.stderr:
            for line in result.stderr.split("\n"):
                if line.strip():
                    logger.info(ansi_escape(line))

        if result.returncode != 0:
            error_msg = f"Apptainer build failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            logger.error(error_msg)
            raise BuildImageError(image_name, error_msg, logger)

        logger.info("Sandbox built successfully!")

    except BuildImageError:
        raise
    except subprocess.TimeoutExpired:
        error_msg = f"Build timed out for {image_name}"
        logger.error(error_msg)
        raise BuildImageError(image_name, error_msg, logger)
    except Exception as e:
        logger.error(f"Error building sandbox {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    finally:
        close_logger(logger)


def build_base_images(
    dataset: list,
    force_rebuild: bool = False,
    namespace: str = None,
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
):
    """
    Build base sandbox images for the dataset.
    """
    test_specs = get_test_specs_from_dataset(
        dataset,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    )
    base_images = {
        x.base_image_key: (x.base_dockerfile, x.platform) for x in test_specs
    }

    for image_name, (dockerfile, platform) in base_images.items():
        sandbox_path = get_sandbox_cache_dir() / sanitize_image_name(image_name)
        if sandbox_path.exists():
            if force_rebuild:
                remove_sandbox(sandbox_path, "quiet")
            else:
                print(f"Base sandbox {image_name} already exists, skipping build.")
                continue

        print(f"Building base sandbox ({image_name})")
        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            build_dir=BASE_IMAGE_BUILD_DIR / sanitize_image_name(image_name),
        )
    print("Base sandboxes built successfully.")


def get_env_configs_to_build(
    dataset: list,
    namespace: str = None,
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
):
    """
    Returns configs for environment images that need to be built.
    """
    image_scripts = dict()
    base_images_checked = set()
    cache_dir = get_sandbox_cache_dir()
    test_specs = get_test_specs_from_dataset(
        dataset,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    )

    for test_spec in test_specs:
        # Check if base sandbox exists
        base_sandbox = cache_dir / sanitize_image_name(test_spec.base_image_key)
        if test_spec.base_image_key not in base_images_checked:
            if not base_sandbox.exists():
                raise Exception(
                    f"Base sandbox {test_spec.base_image_key} not found for {test_spec.env_image_key}\n."
                    "Please build the base images first."
                )
            base_images_checked.add(test_spec.base_image_key)

        # Check if env sandbox already exists
        env_sandbox = cache_dir / sanitize_image_name(test_spec.env_image_key)
        if not env_sandbox.exists():
            image_scripts[test_spec.env_image_key] = {
                "setup_script": test_spec.setup_env_script,
                "dockerfile": test_spec.env_dockerfile,
                "platform": test_spec.platform,
            }
    return image_scripts


def build_env_images(
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
    namespace: str = None,
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
):
    """
    Build environment sandbox images for the dataset.
    """
    cache_dir = get_sandbox_cache_dir()

    if force_rebuild:
        env_image_keys = {
            x.env_image_key
            for x in get_test_specs_from_dataset(
                dataset,
                namespace=namespace,
                instance_image_tag=instance_image_tag,
                env_image_tag=env_image_tag,
            )
        }
        for key in env_image_keys:
            sandbox_path = cache_dir / sanitize_image_name(key)
            remove_sandbox(sandbox_path, "quiet")

    build_base_images(
        dataset, force_rebuild, namespace, instance_image_tag, env_image_tag
    )
    configs_to_build = get_env_configs_to_build(
        dataset, namespace, instance_image_tag, env_image_tag
    )
    if len(configs_to_build) == 0:
        print("No environment sandboxes need to be built.")
        return [], []
    print(f"Total environment sandboxes to build: {len(configs_to_build)}")

    args_list = list()
    for image_name, config in configs_to_build.items():
        args_list.append(
            (
                image_name,
                {"setup_env.sh": config["setup_script"]},
                config["dockerfile"],
                config["platform"],
                ENV_IMAGE_BUILD_DIR / sanitize_image_name(image_name),
            )
        )

    successful, failed = run_threadpool(build_image, args_list, max_workers)
    if len(failed) == 0:
        print("All environment sandboxes built successfully.")
    else:
        print(f"{len(failed)} environment sandboxes failed to build.")

    return successful, failed


def build_instance_images(
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
    namespace: str = None,
    tag: str = "latest",
    env_image_tag: str = "latest",
):
    """
    Build instance sandbox images for the dataset.
    """
    cache_dir = get_sandbox_cache_dir()
    test_specs = list(
        map(
            lambda x: make_test_spec(
                x,
                namespace=namespace,
                instance_image_tag=tag,
                env_image_tag=env_image_tag,
            ),
            dataset,
        )
    )
    if force_rebuild:
        for spec in test_specs:
            sandbox_path = cache_dir / sanitize_image_name(spec.instance_image_key)
            remove_sandbox(sandbox_path, "quiet")

    _, env_failed = build_env_images(
        test_specs, force_rebuild, max_workers
    )

    if len(env_failed) > 0:
        dont_run_specs = [
            spec for spec in test_specs if spec.env_image_key in env_failed
        ]
        test_specs = [
            spec for spec in test_specs if spec.env_image_key not in env_failed
        ]
        print(
            f"Skipping {len(dont_run_specs)} instances - due to failed env sandbox builds"
        )
    print(f"Building instance sandboxes for {len(test_specs)} instances")

    payloads = [(spec, None, False) for spec in test_specs]
    successful, failed = run_threadpool(build_instance_image, payloads, max_workers)
    if len(failed) == 0:
        print("All instance sandboxes built successfully.")
    else:
        print(f"{len(failed)} instance sandboxes failed to build.")

    return successful, failed


def build_instance_image(
    test_spec: TestSpec,
    logger: logging.Logger | None,
    nocache: bool,
):
    """
    Build the instance sandbox for a given test spec.
    """
    cache_dir = get_sandbox_cache_dir()
    build_dir = INSTANCE_IMAGE_BUILD_DIR / sanitize_image_name(
        test_spec.instance_image_key
    )
    new_logger = False
    if logger is None:
        new_logger = True
        logger = setup_logger(test_spec.instance_id, build_dir / "prepare_image.log")

    image_name = test_spec.instance_image_key
    env_image_name = test_spec.env_image_key
    dockerfile = test_spec.instance_dockerfile

    # Check that env sandbox exists
    env_sandbox = cache_dir / sanitize_image_name(env_image_name)
    if not env_sandbox.exists():
        raise BuildImageError(
            test_spec.instance_id,
            f"Environment sandbox {env_image_name} not found for {test_spec.instance_id}",
            logger,
        )

    logger.info(
        f"Environment sandbox {env_image_name} found for {test_spec.instance_id}\n"
        f"Building instance sandbox {image_name} for {test_spec.instance_id}"
    )

    # Check if instance sandbox already exists
    instance_sandbox = cache_dir / sanitize_image_name(image_name)
    if not instance_sandbox.exists():
        build_image(
            image_name=image_name,
            setup_scripts={
                "setup_repo.sh": test_spec.install_repo_script,
            },
            dockerfile=dockerfile,
            platform=test_spec.platform,
            build_dir=build_dir,
            nocache=nocache,
        )
    else:
        logger.info(f"Sandbox {image_name} already exists, skipping build.")

    if new_logger:
        close_logger(logger)


def build_container(
    test_spec: TestSpec,
    run_id: str,
    logger: logging.Logger,
    nocache: bool,
    force_rebuild: bool = False,
) -> Path:
    """
    Build the instance sandbox (if needed) and return its path.

    The sandbox is used read-only at runtime via --writable-tmpfs, so no working
    copy is needed and the original sandbox is never modified.

    Args:
        test_spec: Test specification
        run_id: Run ID
        logger: Logger
        nocache: Whether to force rebuild
        force_rebuild: Whether to force rebuild

    Returns:
        Path to the instance sandbox (read-only, shared across runs)
    """
    cache_dir = get_sandbox_cache_dir()

    # Build instance sandbox if needed
    if force_rebuild:
        sandbox_path = cache_dir / sanitize_image_name(test_spec.instance_image_key)
        remove_sandbox(sandbox_path, "quiet")

    if not test_spec.is_remote_image:
        build_instance_image(test_spec, logger, nocache)
    else:
        # For remote images, pull and convert
        sandbox_path = cache_dir / sanitize_image_name(test_spec.instance_image_key)
        if not sandbox_path.exists():
            logger.info(f"Pulling remote image {test_spec.instance_image_key}...")
            try:
                result = subprocess.run(
                    [
                        "apptainer", "build",
                        "--fakeroot",
                        "--sandbox",
                        str(sandbox_path),
                        f"docker://{test_spec.instance_image_key}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                if result.returncode != 0:
                    raise BuildImageError(
                        test_spec.instance_id, result.stderr, logger
                    )
            except Exception as e:
                raise BuildImageError(
                    test_spec.instance_id, str(e), logger
                ) from e

    instance_sandbox = cache_dir / sanitize_image_name(test_spec.instance_image_key)
    logger.info(f"Using instance sandbox for {test_spec.instance_id}: {instance_sandbox}")
    return instance_sandbox
