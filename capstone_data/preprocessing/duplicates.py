from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import hashlib
import shutil
import numpy as np

from capstone_data.image_loader import(
        get_project_paths,
        load_dataset,
        validate_data_dir,
    )


def _exact_image_hash(image_path):
    '''
    Create a SHA-256 hash from an image's decoded grayscale pixel data.

    Args:
        image_path:
            Path to the image file.

    Returns:
        A hexadecimal SHA-256 hash string.
    '''

    with Image.open(image_path) as image:
        image = image.convert("L")
        pixels = np.asarray(image)

    return hashlib.sha256(pixels.tobytes()).hexdigest()


def handle_duplicates(
    dataset,
    data_dir=None,
    duplicate_dir=None,
):
    """
    Detect and move exact same-class duplicate images, then rebuild
    the dataset index.

    Images are compared using SHA-256 hashes generated from decoded
    grayscale pixel data.

    Same-class duplicates are moved into a separate duplicate directory.

    Cross-class duplicates are treated as label conflicts and are reported
    but not moved automatically.

    Args:
        dataset:
            Dataset records returned by load_dataset().

        data_dir:
            Optional path to the cleaned dataset directory.

            If omitted, the project's standard cleaned-data directory
            is used.

        duplicate_dir:
            Optional directory where duplicate images should be moved.

            If omitted, the project's standard duplicates directory
            is used.

    Returns:
        updated_dataset:
            Rebuilt dataset after duplicate files have been moved.

        results:
            Dictionary containing duplicate-processing information.
    """

    if not dataset:
        raise ValueError(
            "Dataset is empty. Load the dataset before handling duplicates."
        )

    paths = get_project_paths()

    if data_dir is None:
        data_dir = paths["cleaned_data"]

    if duplicate_dir is None:
        duplicate_dir = paths["duplicates"]

    data_dir = Path(data_dir).resolve()
    duplicate_dir = Path(duplicate_dir).resolve()

    validate_data_dir(data_dir)

    if duplicate_dir.is_relative_to(data_dir):
        raise ValueError(
            "duplicate_dir must be outside the cleaned dataset directory."
        )

    duplicate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seen_hashes = {}

    same_class_duplicates = []
    conflicts = []

    print(
        f"Checking {len(dataset)} images for exact duplicates..."
    )

    for item in tqdm(
        dataset,
        desc="Checking duplicates",
    ):
        image_hash = _exact_image_hash(
            item["path"]
        )

        if image_hash not in seen_hashes:
            seen_hashes[image_hash] = item
            continue

        original = seen_hashes[image_hash]

        duplicate_record = {
            "original": original,
            "duplicate": item,
        }

        if original["label"] != item["label"]:
            conflicts.append(
                duplicate_record
            )

        else:
            same_class_duplicates.append(
                duplicate_record
            )

    print()
    print(
        f"Same-class duplicates: {len(same_class_duplicates)}"
    )

    print(
        f"Cross-class conflicts: {len(conflicts)}"
    )

    moved = []

    for record in tqdm(
        same_class_duplicates,
        desc="Moving duplicates",
    ):
        duplicate = record["duplicate"]

        source = Path(
            duplicate["path"]
        )

        class_dir = (
            duplicate_dir
            / duplicate["class_name"]
        )

        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            class_dir
            / source.name
        )

        counter = 1

        while destination.exists():
            destination = (
                class_dir
                / (
                    f"{source.stem}_"
                    f"{counter}"
                    f"{source.suffix}"
                )
            )

            counter += 1

        shutil.move(
            source,
            destination,
        )

        moved.append(
            {
                "original": record["original"],
                "old_path": source,
                "new_path": destination,
            }
        )

    print()
    print(
        f"Duplicates moved: {len(moved)}"
    )

    if conflicts:
        print()
        print(
            "WARNING:",
            len(conflicts),
            "cross-class duplicate conflicts require manual review.",
        )

    print()
    print("Rebuilding dataset...")

    updated_dataset = load_dataset(
        data_dir
    )

    print(
        f"Dataset rebuilt: {len(updated_dataset)} images"
    )

    results = {
        "same_class_duplicates":
            same_class_duplicates,

        "conflicts":
            conflicts,

        "moved":
            moved,
    }

    return updated_dataset, results



if __name__ == "__main__":
    dataset = load_dataset()

    print(f"Total images: {len(dataset)}")
    print(f"First record: {dataset[0]}")
 
