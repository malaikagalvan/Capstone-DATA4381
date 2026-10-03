import os
import numpy as np
import joblib
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from torchvision import transforms

# 1. Data Augmentation Setup
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(48, scale=(0.9, 1.1)),
])

x = []
y = []

# Map emotion folder names to class labels
# 0 = angry, 1 = happy, 2 = neutral, 3 = sad


# Automatically detect the current working directory or absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'MyWorkspaceLP', 'cleaned_data')

classes = [
    os.path.join(data_dir, 'angry_clean'),
    os.path.join(data_dir, 'happy_cleaned'),
    os.path.join(data_dir, 'neutral_cleaned'),
    os.path.join(data_dir, 'sad_cleaned')
]

# 2. Load & Preprocess Images
for label, folder in enumerate(classes):
    if not os.path.exists(folder):
        print(f"Skipping missing directory: {folder}")
        continue

    # Walk through all subfolders (e.g. sad_cleaned_test and sad_cleaned_train)
    for root, dirs, files in os.walk(folder):
        for filename in files:
            img_path = os.path.join(root, filename)
            try:
                # Original Image
                #----------Convert to grayscale----------------------------------------------------#
                img = Image.open(img_path).convert('L') # convert to grayscale
                
                #-------------------Resize to 48x48--------------------------------------------------#
                img_resized = img.resize((48, 48)) # Resize to 48x48
                x.append(np.array(img_resized).flatten())
                y.append(label)

                # Augmented Image
                auger = augment(img)
                auger_resized = auger.resize((48, 48))
                x.append(np.array(auger_resized).flatten())
                y.append(label)
            except Exception:
                pass

X = np.array(x)
Y = np.array(y)

print(f"Loaded dataset: X shape = {X.shape}, Y shape = {Y.shape}")

# 3. Train & Save Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(max_depth=100, n_estimators=10, random_state=42))
])

pipe.fit(X, Y)

#-------------------Joblib example------------------------#
joblib.dump(pipe, 'rf_pipeline_2.joblib')
print("Model trained on all emotions and saved as 'rf_pipeline_2.joblib'!")