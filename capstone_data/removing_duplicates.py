# detecting duplicates
import os
import shutil
from PIL import Image
import imagehash

# Folder containing your cleaned images
clean_dir = ''

# Folder where duplicates will be moved
duplicate_dir = ''

os.makedirs(duplicate_dir, exist_ok=True)

# Store hashes
hash_dict = {}
duplicates_found = 0

for filename in os.listdir(clean_dir):
    path = os.path.join(clean_dir, filename)

    # Skip directories
    if not os.path.isfile(path):
        continue

    try:
        img = Image.open(path)

        # Perceptual hash
        img_hash = imagehash.phash(img)

        if img_hash in hash_dict:
            original = hash_dict[img_hash]

            print(f"\nDuplicate found:")
            print(f"  Original : {original}")
            print(f"  Duplicate: {filename}")

            shutil.move(
                path,
                os.path.join(duplicate_dir, filename)
            )

            duplicates_found += 1

        else:
            hash_dict[img_hash] = filename

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print(f"\nFinished. {duplicates_found} duplicates moved to:")
print(duplicate_dir)
