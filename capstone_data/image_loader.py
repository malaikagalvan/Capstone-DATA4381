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
    }



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
    """Find all model-usable image files inside a class folder."""

    excluded_dirs = {
        "images removed",
        "duplicates_found",
        "duplicates",
    }

    images = []

    for path in folder.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        path_parts = {
            part.lower()
            for part in path.parts
        }

        if path_parts & excluded_dirs:
            continue

        images.append(path)

    if not images:
        raise ValueError(
            f"No supported images found in: {folder}"
        )

    return sorted(images)



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


def load_image(path):
    '''
    Load on FER-2013 image as a normalized NumPy array

    Returns:
        np.ndarray with:
            shape = (48, 48)
            dtype = float32
            values in [0.0, 1.0]

    '''

    path = Path(path)

    with Image.open(path) as image:
        
        image = image.convert("L")

        if image.size != (48, 48):
            raise ValueError(
                    f"Expected 48x48 image, "
                    f"but got {image.size}: {path}"
            )

        image_array = np.asarray(
                image,
                dtype=np.float32,
        )

    image_array /= 255.0

    return image_array


def load_arrays(dataset):
    '''
    Load dataset records into NumPy arrays

    Args:
        dataset:
            Dataset records returned from load_dataset()

    Returns:
        X:
            Image array with shape (N, 48, 48)
            and dtype float32

        y:
            Label array with shape (N,)
            and dtype int64
    '''

    if not dataset:
        raise ValueError(
                "Dataset is empty."
            )

    num_images = len(dataset)

    X = np.empty(
            (num_images, 48, 48),
            dtype=np.float32,
        )

    y = np.empty(
            num_images,
            dtype=np.int64,
        )

    for index, item in enumerate(
            tqdm(
                dataset,
                desc="Loading images nerds",
            )
    ):

        X[index] = load_image(
                item["path"]
        )

        y[index] = item["label"]

    return X, y


if __name__ == "__main__":
    dataset = load_dataset()

    print(f"Total images: {len(dataset)}")
    print(f"First record: {dataset[0]}")
      
