import json
import shutil

from argparse import ArgumentParser
from pathlib import Path

from swebench.harness.apptainer_utils import get_sandbox_work_dir, sanitize_image_name

"""
Script for removing sandbox working copies associated with specified instance IDs.
"""


def main(instance_ids, predictions_path):
    all_ids = set()
    if predictions_path:
        with open(predictions_path, "r") as f:
            predictions = json.loads(f.read())
            for pred in predictions:
                all_ids.add(pred["instance_id"])

    if instance_ids:
        all_ids |= set(instance_ids)

    if not all_ids:
        print("No instance IDs provided, exiting.")
        return

    work_dir = get_sandbox_work_dir()
    for instance_id in all_ids:
        # Look for any working copy matching this instance
        pattern = f"sweb.eval.{instance_id}*"
        found = False
        for entry in work_dir.glob(sanitize_image_name(pattern).replace("*", "*")):
            if entry.is_dir():
                try:
                    shutil.rmtree(str(entry))
                    print(f"Removed sandbox working copy {entry.name}")
                    found = True
                except Exception as e:
                    print(f"Error removing {entry.name}: {e}")
        if not found:
            print(f"No sandbox found for {instance_id}, skipping.")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance_ids",
        help="Instance IDs to remove sandboxes for",
    )
    parser.add_argument(
        "--predictions_path",
        help="Path to predictions file",
    )
    args = parser.parse_args()
    instance_ids = (
        [i.strip() for i in args.instance_ids.split(",")] if args.instance_ids else []
    )
    main(
        instance_ids=instance_ids,
        predictions_path=args.predictions_path,
    )
