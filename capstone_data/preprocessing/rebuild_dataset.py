
from pathlib import Path
from uuid import uuid4
import shutil
import tempfile

from tqdm.auto import tqdm

from capstone_data.image_loader import (
    CLASS_NAMES,
    get_project_paths,
    validate_data_dir,
    find_class_folder,
    find_images,
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
    overwrite=False,
):
    """
    Rebuild a standardized dataset from cleaned images.

    Default source:
        data/cleaned_data/

    Default output:
        data/final/
            angry/
            happy/
            sad/

    Images are copied, not moved.

    Existing populated output directories are protected unless
    overwrite=True is explicitly provided.

    A replacement is prepared in a temporary directory before
    the existing output is replaced.

    Args:
        data_dir:
            Source directory containing cleaned class folders.

        output_dir:
            Destination directory for the standardized dataset.

        overwrite:
            If True, replace an existing output dataset.
            Defaults to False.

    Returns:
        Dictionary with output_dir, total, and class_counts.
    """

    if not isinstance(overwrite, bool):
        raise TypeError(
            "overwrite must be True or False."
        )

    paths = get_project_paths()

    if data_dir is None:
        data_dir = paths["data"] / "cleaned_data"

    if output_dir is None:
        output_dir = paths["data"] / "final"

    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    validate_data_dir(data_dir)

    # Prevent copying a directory into itself, or replacing
    # a parent directory containing the source data.
    if (
        data_dir == output_dir
        or data_dir in output_dir.parents
        or output_dir in data_dir.parents
    ):
        raise ValueError(
            "Source and output directories cannot be "
            "the same or contain one another."
        )

    # --------------------------------------------------------
    # Check existing destination.
    # --------------------------------------------------------

    if output_dir.exists():

        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Expected directory: {output_dir}"
            )

        existing_files = any(
            path.is_file() or path.is_symlink()
            for path in output_dir.rglob("*")
        )

        if existing_files and not overwrite:
            raise FileExistsError(
                f"Output directory already contains files: "
                f"{output_dir}\n"
                "Nothing was changed.\n"
                "Pass overwrite=True to replace it."
            )

    # --------------------------------------------------------
    # Discover and validate all source images first.
    # Reuse the existing image_loader helpers.
    # --------------------------------------------------------

    images_by_class = {}

    for class_name in CLASS_NAMES:

        class_folder = find_class_folder(
            class_name,
            data_dir,
            required=True,
        )

        images_by_class[class_name] = find_images(
            class_folder
        )

    # --------------------------------------------------------
    # Build in a temporary directory beside the destination.
    # This keeps the existing dataset intact during copying.
    # --------------------------------------------------------

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )

    class_counts = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    try:

        for class_name in CLASS_NAMES:

            class_dir = staging_dir / class_name
            class_dir.mkdir()

            for source in tqdm(
                images_by_class[class_name],
                desc=f"Copying {class_name}",
            ):

                destination = _unique_destination(
                    class_dir,
                    source,
                )

                shutil.copy2(
                    source,
                    destination,
                )

                class_counts[class_name] += 1

        # ----------------------------------------------------
        # Copying succeeded. Replace the existing dataset.
        # Temporarily preserve the old output as a backup.
        # ----------------------------------------------------

        backup_dir = None

        if output_dir.exists():

            backup_dir = output_dir.with_name(
                f".{output_dir.name}.backup-{uuid4().hex}"
            )

            if overwrite:
                print(
                    f"\nReplacing existing dataset: {output_dir}"
                )

            output_dir.rename(backup_dir)

        try:
            staging_dir.rename(output_dir)

        except Exception:
            # Restore the original if replacement fails.
            if backup_dir is not None:
                backup_dir.rename(output_dir)
            raise

        # Remove the old dataset only after the new one
        # has been successfully installed.
        if backup_dir is not None:
            try:
                shutil.rmtree(backup_dir)
            except OSError as error:
                print(
                    f"WARNING: Could not remove old backup "
                    f"{backup_dir}: {error}"
                )

    finally:
        # Clean up staging files if anything fails.
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

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

    total = sum(class_counts.values())

    print(f"Total:    {total}")
    print(f"\nOutput: {output_dir}")

    return {
        "output_dir": output_dir,
        "total": total,
        "class_counts": class_counts,
    }

