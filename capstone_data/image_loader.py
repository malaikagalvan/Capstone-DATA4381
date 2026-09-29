from pathlib import Path
from collections import Counter
from PIL import Image
from collections import Counter
import numpy as np
import haslib

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

def validate_data_dir(data_dir=DATA_DIR):
    '''Verify that the dataset directory exists and is not empty.'''

    # Case 1: The path does not exist ya'll.
    if not data_dir.exists():
        raise FileNotFoundError(
                f"Dataset directory not found: {data_dir}\n"
                "Download the cleaned FER-2013 dataset and "
                "place it in the project's data/ directory."
            )

    # Case 2: The path exists but is not a directory.
    if not data_dir.is_dir():
        raise NotADirectoryError(
                f"Expected a directory, but found a file: {data_dir}"
            )

    # Case 3: The directory exists but contains nothing.
    if not any(data_dir.iterdir()):
        raise ValueError(
                f"Dataset directory is empty: {data_dir}\n"
                "Expected four cleaned expression folders."
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
    Discover the cleaned FER-2013 dataset.

    Old train/test dirs are ignored as split boundaries. All images found recursively inside each cleaned class folder are combined into one master dataset.

    Args:
        data_dir:
            Optional path to the dataset directory.

            If omitted, the loader uses the project's default
            data directory.

    Returns:
        A list of dictionaries, one per image.

        Each dictionary contains:
            path       -> Path to the image
            label      -> numeric class label
            class_name -> human-readable class name
    """

    if data_dir is None:
        data_dir = DATA_DIR

    data_dir = Path(data_dir)

    validate_data_dir(data_dir)

    dataset = []

    for class_name in CLASS_NAMES:
        folder = find_class_folder(
            class_name,
            data_dir
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


def exact_image_hash(image_path):
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


def find_exact_duplicates(dataset):
    '''
    Find exact duplicate images across the pooled dataset.

    Duplicate detection is based on decoded grayscale pixel data rather than
    filenames or raw file bytes.

    Args:
        dataset:
            Dataset records returned by load_dataset()

    Returns:
        A list of dictionaries describing duplicate image pairs.
    '''

    seen_hashes = {}
    duplicates = []

    for item in dataset:
        image_hash = exact_image_hash(item["path"])

        if image_hash in seen_hashes:
            original = seen_hashes[image_hash]

            duplicates.append(
                    {
                        "original": original,
                        "duplicate": item,
                        "label_conflict":
                            original["label"] != item["label"],
                    }
                )
        else:
            seen_hashes[image_hash] = item

    return duplicates



if __name__ == "__main__":
    dataset = load_dataset()

    print(f"Total images: {len(dataset)}")
    print(f"First record: {dataset[0]}")
      
