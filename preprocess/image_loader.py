from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np

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
            for path in data_dir.glob(f"{class_name}_cleaned*")
            if path.is_dir()
        )

    if len(matches) != 1:
        raise ValueError(
                f"Expected exactly one folder for '{class_name}', "
                f"but found {len(matches)}: {matches}"
            )

    return matches[0]
        

if __name__ == "__main__":
    validate_data_dir()
    
    # After validating data dir we wil use find_class_folder to verify all class folders.
    for class_name in CLASS_NAMES:
        folder = find_class_folder(class_name)

        print(f"{class_name}: {folder.name}")
        print(f"Label: {CLASS_TO_INDEX[class_name]}")
        print()


