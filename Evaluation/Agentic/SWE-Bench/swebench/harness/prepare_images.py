import resource

from argparse import ArgumentParser

from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.apptainer_build import build_instance_images
from swebench.harness.apptainer_utils import list_sandboxes, get_sandbox_cache_dir, sanitize_image_name
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset, str2bool, optional_str


def filter_dataset_to_build(
    dataset: list,
    instance_ids: list | None,
    force_rebuild: bool,
    namespace: str = None,
    tag: str = None,
    env_image_tag: str = None,
):
    """
    Filter the dataset to only include instances that need to be built.
    """
    cache_dir = get_sandbox_cache_dir()
    existing_sandboxes = list_sandboxes(cache_dir)
    data_to_build = []

    if instance_ids is None:
        instance_ids = [instance[KEY_INSTANCE_ID] for instance in dataset]

    not_in_dataset = set(instance_ids).difference(
        set([instance[KEY_INSTANCE_ID] for instance in dataset])
    )
    if not_in_dataset:
        raise ValueError(f"Instance IDs not found in dataset: {not_in_dataset}")

    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in instance_ids:
            continue

        spec = make_test_spec(
            instance,
            namespace=namespace,
            instance_image_tag=tag,
            env_image_tag=env_image_tag,
        )
        if force_rebuild:
            data_to_build.append(instance)
        elif spec.instance_image_key not in existing_sandboxes:
            data_to_build.append(instance)

    return data_to_build


def main(
    dataset_name,
    split,
    instance_ids,
    max_workers,
    force_rebuild,
    open_file_limit,
    namespace,
    tag,
    env_image_tag,
):
    """
    Build Apptainer sandbox images for the specified instances.
    """
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    dataset = load_swebench_dataset(dataset_name, split)
    dataset = filter_dataset_to_build(
        dataset, instance_ids, force_rebuild, namespace, tag, env_image_tag
    )

    if len(dataset) == 0:
        print("All sandboxes exist. Nothing left to build.")
        return 0

    successful, failed = build_instance_images(
        dataset=dataset,
        force_rebuild=force_rebuild,
        max_workers=max_workers,
        namespace=namespace,
        tag=tag,
        env_image_tag=env_image_tag,
    )
    print(f"Successfully built {len(successful)} sandboxes")
    print(f"Failed to build {len(failed)} sandboxes")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SWE-bench/SWE-bench_Lite",
        help="Name of the dataset to use",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to use")
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max workers for parallel processing"
    )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild images"
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=8192, help="Open file limit"
    )
    parser.add_argument(
        "--namespace",
        type=optional_str,
        default=None,
        help="Namespace to use for the images (default: None)",
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="Tag to use for the images"
    )
    parser.add_argument(
        "--env_image_tag", type=str, default=None, help="Environment image tag to use"
    )
    args = parser.parse_args()
    main(**vars(args))
