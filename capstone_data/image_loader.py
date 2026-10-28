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
    )

# next we will map consistent labels for every model
# this is super important so that the labels don't switch for different models!
CLASS_TO_INDEX = {
        name: index
        for index, name in enumerate(CLASS_NAMES)
        # Therefor {angry: 0, happy: 1, sad: 2,}
    }

IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }


def get_project_paths():
    """
    Return the standard project paths used by the data pipeline.

    Returns:
        A dictionary containing commonly used project directories.

    Raises:
        FileNotFoundError:
            If the project data directory does not exist.

        NotADirectoryError:
            If the data path exists but is not a directory.
    """

    data_dir = ROOT / "data"
    duplicate_dir = data_dir / "duplicates"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but found: {data_dir}"
        )

    duplicate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "root": ROOT,
        "data": data_dir,
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


def find_class_folder(
        class_name,
        data_dir=DATA_DIR,
        required=True,
):

    """
    Find the dir for the given class name.

    Supports both cleaned class dirs such as:
        angry_clean
        angry_cleaned
        angry_cleaned-<timestamp>

    and standard class dirs such as:
        angry

    Args:
        class_name:
            Name of the class to search for.

        data_dir:
            directory containing the class folders.

        required:
            If True, raise an error when the class folder is missing.
            If False, return None when the class folder is missing.

    Returns:
        Path to the class dir, or None if the class is not
        required and is not present.
    """

    data_dir = validate_data_dir(data_dir)

    matches = []

    # first check for an exact folder name:
    # angry/
    exact_folder = data_dir / class_name

    if exact_folder.is_dir():
        matches.append(exact_folder)

    # then check for cleaned folder variants
    cleaned_matches = [
            path
            for path in data_dir.glob(f"{class_name}_clean*")
            if path.is_dir()
    ]

    matches.extend(cleaned_matches)
    
    if len(matches) == 0:
        if required:
            raise ValueError(
                    f"No folder found for '{class_name}'"
                    f"in {data_dir}"
            )

        return None

    if len(matches) > 1:
        raise ValueError(
                f"Expected one folder for '{class_name}', "
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

def list_datasets():
    """
    Discover available dataset directories inside the project's data folder.

    A dataset directory must contain at least one recognized class folder
    directly inside it.

    Returns:
        A sorted list of available dataset names.
    """

    paths = get_project_paths()
    data_dir = paths["data"]

    datasets = []

    for path in data_dir.iterdir():

        if not path.is_dir():
            continue

        if path.name == "duplicates":
            continue

        has_class_folder = any(
            (path / class_name).is_dir()
            for class_name in CLASS_NAMES
        )

        if has_class_folder:
            datasets.append(
                path.name
            )

    return sorted(datasets)

def load_dataset(
    datasets="final",
):
    """
    Discover images from one or more selected datasets and return a
    Dataset sources may contain either cleaned class directories such as:

        angry_cleaned/
        happy_cleaned/

    or standard class directories such as:

        angry/
        happy/

    Old train/test directories inside class folders are ignored as split
    boundaries because image discovery is recursive.

    Args:
        datasets:
            Name of a dataset directory or a list of dataset directories
            located inside the project's data directory.

            Examples:
                "cleaned_data"

                "aug_data_v1"

                [
                    "cleaned_data",
                    "aug_data_v1",
                    "aug_data_v2",
                ]

    Returns:
        A list of dictionaries, one per image.

        Each dictionary contains:
            path       -> Path to the image
            label      -> numeric class label
            class_name -> human-readable class name
            source     -> dataset the image came from
    """

    paths = get_project_paths()
    data_root = paths["data"]

    # Allow one dataset name to be supplied as a string.
    if isinstance(datasets, str):
        datasets = [datasets]

    # Make sure a valid collection was supplied.
    if not isinstance(datasets, (list, tuple)):
        raise TypeError(
            "datasets must be a dataset name or "
            "a list/tuple of dataset names."
        )

    if not datasets:
        raise ValueError(
            "At least one dataset must be selected."
        )

    # Remove repeated dataset names while keeping their order.
    datasets = list(
        dict.fromkeys(datasets)
    )

    available_datasets = list_datasets()

    dataset = []

    for source_name in datasets:

        if source_name not in available_datasets:
            raise ValueError(
                f"Unknown dataset '{source_name}'. "
                f"Available datasets: "
                f"{available_datasets}"
            )

        source_dir = (
            data_root
            / source_name
        )

        validate_data_dir(
            source_dir
        )

        # The original cleaned dataset should contain
        # every class.
        #
        # Augmented datasets are allowed to contain only
        # selected classes.
        require_all_classes = (
            source_name == "final"
        )

        for class_name in CLASS_NAMES:

            folder = find_class_folder(
                class_name,
                source_dir,
                required=require_all_classes,
            )

            # Augmentation datasets may intentionally
            # omit certain classes, such as neutral.
            if folder is None:
                continue

            images = find_images(
                folder
            )

            label = CLASS_TO_INDEX[
                class_name
            ]

            for image_path in images:

                dataset.append(
                    {
                        "path": image_path,
                        "label": label,
                        "class_name": class_name,
                        "source": source_name,
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


def stratified_split_indices(
    y,
    train_ratio=0.80,
    val_ratio=0.10,
    test_ratio=0.10,
    seed=42,
):
    """
    Create reproducible, stratified train/validation/test indices.

    Validation is optional. Set val_ratio=0.0 to create
    only train and test splits.

    Args:
        y:
            One-dimensional NumPy array of class labels.

        train_ratio:
            Fraction of each class used for training.

        val_ratio:
            Fraction of each class used for validation.
            May be 0.0.

        test_ratio:
            Fraction of each class used for testing.

        seed:
            Seed used by NumPy's random-number generator.

    Returns:
        train_indices
        val_indices
        test_indices
    """

    if y.ndim != 1:
        raise ValueError(
            f"Expected y to be one-dimensional, "
            f"but got shape {y.shape}"
        )

    if len(y) == 0:
        raise ValueError(
            "Cannot split an empty dataset."
        )

    ratios = (
        train_ratio,
        val_ratio,
        test_ratio,
    )

    if any(ratio < 0 for ratio in ratios):
        raise ValueError(
            "Split ratios cannot be negative."
        )

    if train_ratio <= 0:
        raise ValueError(
            "train_ratio must be greater than zero."
        )

    if test_ratio <= 0:
        raise ValueError(
            "test_ratio must be greater than zero."
        )

    if not np.isclose(
        train_ratio + val_ratio + test_ratio,
        1.0,
    ):
        raise ValueError(
            "train_ratio + val_ratio + test_ratio "
            "must equal 1.0."
        )

    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    classes = np.unique(y)

    for class_label in classes:

        class_indices = np.where(
            y == class_label
        )[0]

        shuffled_indices = rng.permutation(
            class_indices
        )

        num_samples = len(
            shuffled_indices
        )

        num_train = int(
            num_samples * train_ratio
        )

        num_val = int(
            num_samples * val_ratio
        )

        num_test = (
            num_samples
            - num_train
            - num_val
        )

        if num_train == 0:
            raise ValueError(
                f"Class {class_label} has too few samples "
                "for the training split."
            )

        if val_ratio > 0 and num_val == 0:
            raise ValueError(
                f"Class {class_label} has too few samples "
                "for the validation split."
            )

        if test_ratio > 0 and num_test == 0:
            raise ValueError(
                f"Class {class_label} has too few samples "
                "for the test split."
            )

        train_end = num_train
        val_end = num_train + num_val

        train_indices.extend(
            shuffled_indices[:train_end]
        )

        val_indices.extend(
            shuffled_indices[
                train_end:val_end
            ]
        )

        test_indices.extend(
            shuffled_indices[val_end:]
        )

    # IMPORTANT:
    # Everything below this point is OUTSIDE the class loop.

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64,
    )

    val_indices = np.asarray(
        val_indices,
        dtype=np.int64,
    )

    test_indices = np.asarray(
        test_indices,
        dtype=np.int64,
    )

    train_indices = rng.permutation(
        train_indices
    )

    val_indices = rng.permutation(
        val_indices
    )

    test_indices = rng.permutation(
        test_indices
    )

    return (
        train_indices,
        val_indices,
        test_indices,
    )                  

def split_dataset(
        X,
        y,
        train_ratio=0.70,
        val_ratio=0.00,
        test_ratio=0.30,
        seed=42,
    ):
    '''
    Split X and y into stratified train, validation,
    and test sets.

    Set val_ratio=0.00 to disable validation.

    Returns:
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    '''

    if len(X) != len(y):
        raise ValueError(
                "X and y must contain the same number of samples."
            )

    (
        train_indices,
        val_indices,
        test_indices,
    ) = stratified_split_indices(
            y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )






    
if __name__ == "__main__":
    dataset = load_dataset()

    print(f"Total images: {len(dataset)}")
    print(f"First record: {dataset[0]}")

