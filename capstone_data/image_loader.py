from tqdm.auto import tqdm
from pathlib import Path
from collections import Counter
from PIL import Image

import numpy as np
import hashlib
import shutil

ROOT = Path(__file__).resolve().parents[1]  # __file__ refers to YOUR current Python module
# file is converted into a path object, resolve gives abs path and parents[1] moves us out 
# to the proper 'Capstone-DATA4381' directory. From here we can verify it exists and we are setup properly. 

DATA_DIR = ROOT / "data" # the divisor element for a path object joins 'data'

CLASS_NAMES = (
        "angry",
        "happy",
        "sad",
        "neutral",
    )

# next we will map consistent labels for every model
# this is super important so that the labels don't switch for different models!
CLASS_TO_INDEX = {
        name: index
        for index, name in enumerate(CLASS_NAMES)
        # Therefor {angry: 0, happy: 1, sad: 2, neutral: 3}
    }

IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }


def get_project_paths():
    '''
    Return and validate the standard project paths used by the 
    data pipeline.

    Returns:
        A dictionary containing commonly used project dirs

    Raises:
        FileNotFoundError:
            If the expected data or cleaned-data dir does not exist

        NotADirectoryError:
            If an expected dir path exists but is not a dir
    '''

    data_dir = ROOT / "data"
    cleaned_data_dir = data_dir / "cleaned_data"
    duplicate_dir = data_dir / "duplicates"

    if not data_dir.exists():
        raise FileNotFoundError(
                f"Data directory not found: {data_dir}"
            )

    if not data_dir.is_dir():
        raise NotADirectoryError(
                f"Expected a directory but found: {data_dir}"
            )

    if not cleaned_data_dir.exists():
        raise FileNotFoundError(
                f"Cleaned dataset directory not found: {cleaned_data_dir}"
            )

    if not cleaned_data_dir.is_dir():
        raise NotADirectoryError(
                f"Expected a directory but found: {cleaned_data_dir}"
            )

    duplicate_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    return {
        "root": ROOT,
        "data": data_dir,
        "cleaned_data": cleaned_data_dir,
        "duplicates": duplicate_dir,
    )



def validate_data_dir(data_dir=DATA_DIR):
    '''Verify that the dataset directory exists and is not empty.'''

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
                f"Data directory not found: {data_dir}"
            )

    if not data_dir.is_dir():
        raise NotADirectoryError(
                f"Expected a directory but found: {data_dir}"
            )

    if not any(data_dir.iterdir()):
        raise ValueError(
            f"Data directory is empty: {data_dir}"
        )

    return data_dir


def find_class_folder(class_name, data_dir=DATA_DIR):
    '''Find the cleaned directory by class_name'''

    data_dir = validate_data_dir(data_dir) # first validate the data dir exists

    # next we are going to match class names (i.e angry) and * with glob() means any
    # variation of this dir will suffice. This is useful because each person downloading
    # will likely have unique timestamps in their directory names (angry_cleaned-20260927T154645Z-1-001)
    # We then create local paths and verify they are directories.
    matches = sorted(
            path
            for path in data_dir.glob(f"{class_name}_clean*")
            if path.is_dir()
        )

    if len(matches) != 1:
        raise ValueError(
                f"Expected exactly one folder for '{class_name}', "
                f"but found {len(matches)}: {matches}"
            )

    return matches[0]
        

def find_images(folder):
    '''Find all supported image files inside a class folder.'''

    images = []

    for path in folder.rglob("*"):  #rglob is the same as glob but recursive (it will search
        # directories and all subdirectories)

        if path.is_file():

            extension = path.suffix.lower()  # return just extension and lower to match 
            # IMAGE_EXTENSIONS format

            if extension in IMAGE_EXTENSIONS:  # if our selected extension is an IMAGE_EXTENSION
                # then add it to image list
                images.append(path)

    if not images:
        raise ValueError(
                f"No supported images found in: {folder}"
            )

    return sorted(images)  # sort for reproducibility


def find_split_folder(class_folder, class_name, split):
    '''Locate a class's cleaned train or test directory.'''

    # verify requested split is valid name
    split = split.lower()

    if split not in ("train", "test"):
        raise ValueError(
                f"Invalid split: '{split}'. Expected 'train' or 'test'."
            )

    matches = []

    # search all nested directories
    for path in class_folder.rglob("*"):  # similar to find_class_folder()

        if not path.is_dir():
            continue

        # force proper naming convention
        folder_name = path.name.lower()

        # check for train* or test* in directory name (AKA all variants)
        parts = folder_name.split("_")

        if any(part.startswith(split) for part in parts):
            matches.append(path)

    if len(matches) != 1:
        raise ValueError(
                f"Expected exactly one '{split}' directory for "
                f"'{class_name}', but found {len(matches)}: {matches}"
            )

    return matches[0]


def load_dataset(data_dir=None):
    """
    Discover all cleaned FER-2013 images and return a pooled dataset index.

    Old train/test directories are ignored as split boundaries. All images
    found recursively inside each cleaned class folder are combined into one
    master dataset.

    Args:
        data_dir:
            Optional path to the cleaned dataset directory.

            If omitted, the project's standard cleaned-data directory
            is used.

    Returns:
        A list of dictionaries, one per image.

        Each dictionary contains:
            path       -> Path to the image
            label      -> numeric class label
            class_name -> human-readable class name
    """

    if data_dir is None:
        paths = get_project_paths()
        data_dir = paths["cleaned_data"]

    data_dir = Path(data_dir)

    validate_data_dir(data_dir)

    dataset = []

    for class_name in CLASS_NAMES:
        folder = find_class_folder(
            class_name,
            data_dir,
        )

        images = find_images(folder)

        label = CLASS_TO_INDEX[class_name]

        for image_path in images:
            dataset.append(
                {
                    "path": image_path,
                    "label": label,
                    "class_name": class_name,
                }
            )

    return dataset



def count_images_by_class(dataset):
    """
    Count the number of images in each class.

    Args:
        dataset:
            Dataset records returned from load_dataset()

    Returns:
        A counter mapping class names to image counts
    """

    return Counter(
            item["class_name"]
            for item in dataset
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
      
