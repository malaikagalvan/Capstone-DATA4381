from pathlib import Path

import shutil

from tqdm.auto import tqdm

from capstone_data.image_loader import (
    CLASS_NAMES,
    get_project_paths,
    load_dataset,
)


def _unique_destination(class_dir, source):
    """
    Create a destination path without overwriting an existing file.

    Example:

        image.jpg
        image_1.jpg
        image_2.jpg
    """

    destination = class_dir / source.name

    counter = 1

    while destination.exists():

        destination = (
            class_dir
            / f"{source.stem}_{counter}{source.suffix}"
        )

        counter += 1

    return destination


def rebuild_dataset(
    data_dir=None,
    output_dir=None,
):
    """
    Rebuild the cleaned FER-2013 dataset into one standardized
    directory structure.

    Output:

        data/
            final/
                angry/
                happy/
                sad/
                neutral/

    Source images are copied, not moved.

    Args:
        data_dir:
            Optional cleaned-data directory.

            If omitted, the project's standard cleaned-data
            directory is used.

        output_dir:
            Optional output directory.

            If omitted, data/final/ is used.

    Returns:
        Dictionary containing the output path, total number
        of copied images, and copied count for each class.
    """
    if data_dir is None:
        paths = get_project_paths()
        data_dir = paths["cleaned_data"]

    if output_dir is None:
        output_dir = Path(data_dir).parent / "final"


    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    dataset = load_dataset(data_dir)

    # --------------------------------------------------------
    # Protect against accidentally rebuilding on top of an
    # already-populated final dataset.
    # --------------------------------------------------------

    if output_dir.exists():

        existing_files = [
            path
            for path in output_dir.rglob("*")
            if path.is_file()
        ]

        if existing_files:
            raise FileExistsError(
                f"Output directory already contains files: "
                f"{output_dir}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create standardized class directories.
    # --------------------------------------------------------

    for class_name in CLASS_NAMES:

        class_dir = output_dir / class_name

        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Track how many images are copied per class.
    # --------------------------------------------------------

    class_counts = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    # --------------------------------------------------------
    # Copy every usable image into its standardized class dir.
    # --------------------------------------------------------

    for item in tqdm(
        dataset,
        desc="Rebuilding dataset",
    ):

        source = Path(
            item["path"]
        )

        class_name = item[
            "class_name"
        ]

        class_dir = (
            output_dir
            / class_name
        )

        destination = _unique_destination(
            class_dir,
            source,
        )

        shutil.copy2(
            source,
            destination,
        )

        class_counts[class_name] += 1

    # --------------------------------------------------------
    # Report results.
    # --------------------------------------------------------

    print()
    print("Dataset rebuild complete")
    print("=" * 40)

    for class_name in CLASS_NAMES:

        print(
            f"{class_name:8s}: "
            f"{class_counts[class_name]}"
        )

    print("-" * 40)
    print(
        f"Total:    "
        f"{sum(class_counts.values())}"
    )

    print(
        f"\nOutput: {output_dir}"
    )

    return {
        "output_dir": output_dir,
        "total": sum(class_counts.values()),
        "class_counts": class_counts,
    }
