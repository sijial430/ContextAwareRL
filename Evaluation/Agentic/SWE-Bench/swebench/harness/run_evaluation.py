from __future__ import annotations

import json
import platform
import threading
import traceback

if platform.system() == "Linux":
    import resource

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path, PurePosixPath
from tqdm.auto import tqdm

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
)
from swebench.harness.apptainer_utils import (
    clean_sandboxes,
    exec_run_with_timeout,
    get_sandbox_cache_dir,
    list_sandboxes,
    remove_sandbox,
    sanitize_image_name,
    should_remove,
)
from swebench.harness.apptainer_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.reporting import make_run_report
from swebench.harness.modal_eval import (
    run_instances_modal,
    validate_modal_credentials,
)
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import (
    EvaluationError,
    load_swebench_dataset,
    get_predictions_from_file,
    run_threadpool,
    str2bool,
    optional_str,
)

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def run_instance(
    test_spec: TestSpec,
    pred: dict,
    rm_image: bool,
    force_rebuild: bool,
    run_id: str,
    timeout: int | None = None,
    rewrite_reports: bool = False,
) -> dict:
    """
    Run a single instance with the given prediction using Apptainer sandboxes.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        run_id (str): Run ID
        timeout (int): Timeout for running tests
        rewrite_reports (bool): True if eval run is just to reformat existing report
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id

    # Set up report file
    report_path = log_dir / LOG_REPORT
    if rewrite_reports:
        test_output_path = log_dir / LOG_TEST_OUTPUT
        if not test_output_path.exists():
            raise ValueError(f"Test output file {test_output_path} does not exist")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return {
            "completed": True,
            "resolved": report[instance_id]["resolved"],
        }
    if report_path.exists():
        report = json.loads(report_path.read_text())
        return {
            "completed": True,
            "resolved": report[instance_id]["resolved"],
        }

    if not test_spec.is_remote_image:
        build_dir = INSTANCE_IMAGE_BUILD_DIR / sanitize_image_name(
            test_spec.instance_image_key
        )
        image_build_link = log_dir / "image_build_dir"
        if not image_build_link.exists():
            try:
                image_build_link.symlink_to(
                    build_dir.absolute(), target_is_directory=True
                )
            except:
                pass

    # Set up logger
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)

    # Run the instance
    sandbox_path = None
    eval_completed = False
    report = {}
    try:
        # Build sandbox (shared, read-only via --writable-tmpfs at runtime)
        sandbox_path = build_container(
            test_spec, run_id, logger, rm_image, force_rebuild
        )
        logger.info(f"Sandbox for {instance_id} ready: {sandbox_path}")

        # Write patch and eval script to host log_dir, bind-mount into container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred[KEY_PREDICTION] or "")
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to sandbox..."
        )

        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)

        # Bind-mount host files into the container (read-only from host)
        # The container uses --writable-tmpfs so all writes go to a tmpfs overlay.
        # Because each apptainer exec is a separate process with its own overlay,
        # we must run apply-patch + eval in a single exec call to preserve state.
        bind_mounts = [
            (str(patch_file), str(DOCKER_PATCH)),
            (str(eval_file), "/eval.sh"),
            (str(log_dir), "/tmp/log_dir"),
        ]

        # Build a combined shell script that:
        # 1. Applies patch (tries multiple methods)
        # 2. Records git diff before eval
        # 3. Runs eval script
        # 4. Records git diff after eval
        # 5. Writes outputs to bind-mounted log_dir
        apply_cmds = []
        for i, git_apply_cmd in enumerate(GIT_APPLY_CMDS):
            if i == 0:
                apply_cmds.append(
                    f'if {git_apply_cmd} {DOCKER_PATCH}; then echo "APPLY_PATCH_SUCCESS"; APPLIED=1'
                )
            else:
                apply_cmds.append(
                    f'elif {git_apply_cmd} {DOCKER_PATCH}; then echo "APPLY_PATCH_SUCCESS"; APPLIED=1'
                )
        apply_cmds.append('else echo "APPLY_PATCH_FAILURE"; APPLIED=0; fi')
        apply_block = "\n".join(apply_cmds)

        combined_script = f"""set -e
cd {DOCKER_WORKDIR}

# Step 1: Apply patch
APPLIED=0
{apply_block}

if [ "$APPLIED" -eq 0 ]; then
    echo "APPLY_PATCH_FAIL"
    exit 1
fi

# Step 2: Git diff before eval
echo "==DIFF_BEFORE_START=="
git -c core.fileMode=false diff
echo "==DIFF_BEFORE_END=="

# Step 3: Run eval (don't exit on failure, capture output)
set +e
/bin/bash /eval.sh > /tmp/log_dir/test_output.txt 2>&1
EVAL_EXIT=$?
set -e

# Step 4: Git diff after eval
echo "==DIFF_AFTER_START=="
git -c core.fileMode=false diff
echo "==DIFF_AFTER_END=="

exit $EVAL_EXIT
"""
        combined_script_file = Path(log_dir / "combined_eval.sh")
        combined_script_file.write_text(combined_script)
        bind_mounts.append((str(combined_script_file), "/combined_eval.sh"))

        # Run everything in a single apptainer exec (single tmpfs overlay)
        output, timed_out, total_runtime = exec_run_with_timeout(
            sandbox_path,
            "/bin/bash /combined_eval.sh",
            timeout,
            workdir=DOCKER_WORKDIR,
            bind_mounts=bind_mounts,
        )

        logger.info(f"Test runtime: {total_runtime:_.2f} seconds")

        # Check if patch was applied
        if "APPLY_PATCH_SUCCESS" not in output:
            logger.info(f"{APPLY_PATCH_FAIL}:\n{output}")
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{output}",
                logger,
            )
        logger.info(f"{APPLY_PATCH_PASS}")

        # Extract git diff before eval
        diff_before = ""
        if "==DIFF_BEFORE_START==" in output and "==DIFF_BEFORE_END==" in output:
            diff_before = output.split("==DIFF_BEFORE_START==")[1].split("==DIFF_BEFORE_END==")[0].strip()
        logger.info(f"Git diff before:\n{diff_before}")

        # Extract git diff after eval
        diff_after = ""
        if "==DIFF_AFTER_START==" in output and "==DIFF_AFTER_END==" in output:
            diff_after = output.split("==DIFF_AFTER_START==")[1].split("==DIFF_AFTER_END==")[0].strip()
        logger.info(f"Git diff after:\n{diff_after}")
        if diff_after != diff_before:
            logger.info("Git diff changed after running eval script")

        # Read test output from bind-mounted log_dir
        test_output_path = log_dir / LOG_TEST_OUTPUT
        test_output_from_bind = log_dir / "test_output.txt"
        if test_output_from_bind.exists():
            test_output = test_output_from_bind.read_text()
            with open(test_output_path, "w") as f:
                f.write(test_output)
                if timed_out:
                    f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
        else:
            # Fallback: use combined output
            test_output = output
            with open(test_output_path, "w") as f:
                f.write(test_output)

        logger.info(f"Test output for {instance_id} written to {test_output_path}")

        if timed_out:
            raise EvaluationError(
                instance_id,
                f"Test timed out after {timeout} seconds.",
                logger,
            )

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
        )

        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        eval_completed = True
    except (EvaluationError, BuildImageError) as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({logger.log_file}) for more information."
        )
        logger.error(error_msg)
    finally:
        # No working copy to clean up — sandbox was used read-only via --writable-tmpfs
        if rm_image:
            # Remove the cached instance sandbox
            cache_dir = get_sandbox_cache_dir()
            instance_sandbox = cache_dir / sanitize_image_name(
                test_spec.instance_image_key
            )
            remove_sandbox(instance_sandbox, logger)
        close_logger(logger)
        return {
            "completed": eval_completed,
            "resolved": report.get(instance_id, {}).get("resolved", False),
        }


def run_instances(
    predictions: dict,
    instances: list,
    cache_level: str,
    clean: bool,
    force_rebuild: bool,
    max_workers: int,
    run_id: str,
    timeout: int,
    namespace: str | None = "swebench",
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
    rewrite_reports: bool = False,
):
    """
    Run all instances for the given predictions in parallel.
    """
    cache_dir = get_sandbox_cache_dir()
    test_specs = list(
        map(
            lambda instance: make_test_spec(
                instance,
                namespace=namespace,
                instance_image_tag=instance_image_tag,
                env_image_tag=env_image_tag,
            ),
            instances,
        )
    )

    # Check existing instance sandboxes
    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_sandboxes = list_sandboxes(cache_dir)
    existing_images = {
        name for name in existing_sandboxes if name in instance_image_ids
    }
    if not force_rebuild and len(existing_images):
        print(
            f"Found {len(existing_images)} existing instance sandboxes. Will reuse them."
        )

    # Build payloads
    payloads = []
    for test_spec in test_specs:
        payloads.append(
            (
                test_spec,
                predictions[test_spec.instance_id],
                should_remove(
                    test_spec.instance_image_key,
                    cache_level,
                    clean,
                    existing_images,
                ),
                force_rebuild,
                run_id,
                timeout,
                rewrite_reports,
            )
        )

    print(f"Running {len(instances)} instances...")
    stats = {"✓": 0, "✖": 0, "error": 0}
    pbar = tqdm(total=len(payloads), desc="Evaluation", postfix=stats)
    lock = threading.Lock()

    def run_evaluation_with_progress(*args):
        result = run_instance(*args)
        with lock:
            if result["completed"]:
                if result["resolved"]:
                    stats["✓"] += 1
                else:
                    stats["✖"] += 1
            else:
                stats["error"] += 1
            pbar.set_postfix(stats)
            pbar.update()
        return result

    run_threadpool(run_evaluation_with_progress, payloads, max_workers)
    print("All instances run.")


def get_dataset_from_preds(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions: dict,
    run_id: str,
    rewrite_reports: bool,
    exclude_completed: bool = True,
):
    """
    Return only instances that have predictions and are in the dataset.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    dataset = load_swebench_dataset(dataset_name, split)
    dataset_ids = {i[KEY_INSTANCE_ID] for i in dataset}

    if instance_ids:
        missing_preds = set(instance_ids) - set(predictions.keys())
        if missing_preds:
            print(
                f"Warning: Missing predictions for {len(missing_preds)} instance IDs."
            )

    prediction_ids = set(predictions.keys())
    if prediction_ids - dataset_ids:
        raise ValueError(
            (
                "Some prediction IDs not found in dataset!"
                f"\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}"
            )
        )
    if instance_ids:
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in instance_ids]

    if rewrite_reports:
        test_output_ids = set()
        for instance in dataset:
            if instance[KEY_INSTANCE_ID] not in predictions:
                continue
            prediction = predictions[instance[KEY_INSTANCE_ID]]
            test_output_file = (
                RUN_EVALUATION_LOG_DIR
                / run_id
                / prediction["model_name_or_path"].replace("/", "__")
                / prediction[KEY_INSTANCE_ID]
                / "test_output.txt"
            )
            if test_output_file.exists():
                test_output_ids.add(instance[KEY_INSTANCE_ID])
        dataset = [
            i
            for i in dataset
            if i[KEY_INSTANCE_ID] in prediction_ids
            and i[KEY_INSTANCE_ID] in test_output_ids
        ]
        return dataset

    completed_ids = set()
    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in prediction_ids:
            continue
        prediction = predictions[instance[KEY_INSTANCE_ID]]
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction[KEY_MODEL].replace("/", "__")
            / prediction[KEY_INSTANCE_ID]
            / LOG_REPORT
        )
        if report_file.exists():
            completed_ids.add(instance[KEY_INSTANCE_ID])

    if completed_ids and exclude_completed:
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] not in completed_ids]

    empty_patch_ids = {
        k
        for k, v in predictions.items()
        if v[KEY_PREDICTION] == "" or v[KEY_PREDICTION] is None
    }

    dataset = [
        i
        for i in dataset
        if i[KEY_INSTANCE_ID] in prediction_ids
        and i[KEY_INSTANCE_ID] not in empty_patch_ids
    ]
    return dataset


def main(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions_path: str,
    max_workers: int,
    force_rebuild: bool,
    cache_level: str,
    clean: bool,
    open_file_limit: int,
    run_id: str,
    timeout: int,
    namespace: str | None,
    rewrite_reports: bool,
    modal: bool,
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
    report_dir: str = ".",
):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    if dataset_name == "SWE-bench/SWE-bench_Multimodal" and split == "test":
        print(
            "⚠️ Local evaluation for the test split of SWE-bench Multimodal is not supported. "
            "Please check out sb-cli (https://github.com/swe-bench/sb-cli/) for instructions on how to submit predictions."
        )
        return

    assert len(run_id) > 0, "Run ID must be provided"
    if report_dir is not None:
        report_dir = Path(report_dir)
        if not report_dir.exists():
            report_dir.mkdir(parents=True)

    if force_rebuild and namespace is not None:
        raise ValueError("Cannot force rebuild and use a namespace at the same time.")

    predictions = get_predictions_from_file(predictions_path, dataset_name, split)
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    dataset = get_dataset_from_preds(
        dataset_name, split, instance_ids, predictions, run_id, rewrite_reports
    )
    full_dataset = load_swebench_dataset(dataset_name, split, instance_ids)

    if modal:
        if not dataset:
            print("No instances to run.")
        else:
            validate_modal_credentials()
            run_instances_modal(predictions, dataset, full_dataset, run_id, timeout)
        return

    # Run instances locally with Apptainer
    if platform.system() == "Linux":
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    cache_dir = get_sandbox_cache_dir()
    existing_images = list_sandboxes(cache_dir)

    if not dataset:
        print("No instances to run.")
    else:
        # Build environment sandboxes + run instances
        if namespace is None and not rewrite_reports:
            build_env_images(
                dataset,
                force_rebuild,
                max_workers,
                namespace,
                instance_image_tag,
                env_image_tag,
            )
        run_instances(
            predictions,
            dataset,
            cache_level,
            clean,
            force_rebuild,
            max_workers,
            run_id,
            timeout,
            namespace=namespace,
            instance_image_tag=instance_image_tag,
            env_image_tag=env_image_tag,
            rewrite_reports=rewrite_reports,
        )

    # Clean sandboxes + make final report
    clean_sandboxes(cache_dir, existing_images, cache_level, clean)
    return make_run_report(
        predictions,
        full_dataset,
        run_id,
        namespace,
        instance_image_tag,
        env_image_tag,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run evaluation harness for the given dataset and predictions.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Common args
    parser.add_argument(
        "-d",
        "--dataset_name",
        default="SWE-bench/SWE-bench_Lite",
        type=str,
        help="Name of dataset or path to JSON file.",
    )
    parser.add_argument(
        "-s", "--split", type=str, default="test", help="Split of the dataset"
    )
    parser.add_argument(
        "-i",
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "-p",
        "--predictions_path",
        type=str,
        help="Path to predictions file - if 'gold', uses gold predictions",
        required=True,
    )

    # Local execution args
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of workers (should be <= 75%% of CPU cores)",
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=4096, help="Open file limit"
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=1_800,
        help="Timeout (in seconds) for running tests for each instance",
    )
    parser.add_argument(
        "--force_rebuild",
        type=str2bool,
        default=False,
        help="Force rebuild of all images",
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument(
        "-id", "--run_id", type=str, required=True, help="Run ID - identifies the run"
    )
    parser.add_argument(
        "-n",
        "--namespace",
        type=optional_str,
        default="swebench",
        help='Namespace for images. (use "none" to use no namespace)',
    )
    parser.add_argument(
        "--instance_image_tag", type=str, default="latest", help="Instance image tag"
    )
    parser.add_argument(
        "--env_image_tag", type=str, default="latest", help="Environment image tag"
    )
    parser.add_argument(
        "--rewrite_reports",
        type=str2bool,
        default=False,
        help="Doesn't run new instances, only writes reports for instances with existing test outputs",
    )
    parser.add_argument(
        "--report_dir", type=str, default=".", help="Directory to write reports to"
    )

    # Modal execution args
    parser.add_argument("--modal", type=str2bool, default=False, help="Run on Modal")

    args = parser.parse_args()
    main(**vars(args))
