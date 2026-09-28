## Image Loader Documentation

Hey ya'll :) Here's how the image_loader module works so that way our data and image loading is consistent for the full capstone from here on out. 

1. Download the cleaned dataset files from the shared notion for the group
<p align="center">
  <img src="../misc/doc_pics/data_clean.png" width="300" height="300" alt="Data cleaning">
</p>

2. Unzip the files and place them in a folder (i use the github data folder once it's been cloned but you don't have to)
<div style="display: flex; justify-content: space-between; align-items: center;">
  <img src="../misc/doc_pics/extract_data.png" width="300" alt="Extract data">
  <img src="../misc/doc_pics/data_extracted.png" width="400" alt="Extracted data">
</div>

3. Next from command line run this command 
```cmd
python -m pip install git+https://github.com/malaikagalvan/Capstone-DATA4381.git
```

4. Now that we have our module installed in any jupyter notebook or py code just add

```python
from capstone_data.image_loader import load_dataset

dataset = load_dataset(r"C:\path to your cleaned_data folder")
```

If you see this output you have everything loaded and ready to go!
