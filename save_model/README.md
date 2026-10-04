# **How to save your model**  

### **Make sure your model has**  

1. **All Images Converted to Greyscale using .convert('L'):**  


    *example->  
     img = Image.open(img_path).convert('L')*

2. **All Images sized to 48x48 .resize((48,48)):**  
    
    *example ->  
    img_resized = img.resize((48, 48))*  

---
### **Saving the Model**  

1) **Install Joblib**  (skip this if you already have scikit learn installed) 

    pip install joblib
  
  
2) **At the very end of your code:** 

    joblib.dump(your_pipeline_name, 'name_for_file.joblib')    



**Note:** It may take a while for the it to excute the joblib file. When it is done you will see a new *name_for_file.joblib* file.

**Please upload the final joblib file to the repository :)**



*example code located in: save_model/baseline_draft_rf.py*


Note from Sheehey

Because the NN is from scratch its a little different to save the model because of all the from scratch configurations

1. Make sure you are in the environment that works with the repo
   ```
   python -m pip install -e .
   ```

2. Then run the loader
   ```
   python saved_models/load_nn_baseline.py
   ```
You should see something like
   Neural network baseline loaded successfully.

   classes:
   0: angry
   1: happy
   2: sad
   3: neutral

   Training configuration:
   epochs: 30
   batch_size: 64
   learning_rate: 0.01
   ...

3. If you want to use the baseline in your own code or notebook heres what you can do
```
from saved_models.load_nn_baseline import (
    load_nn_baseline,
    predict,
)


saved_model = load_nn_baseline()

model = saved_model["model"]

class_names = saved_model[
    "class_names"
]


predictions, probabilities = predict(
    model,
    X_test,
)


print(
    predictions[:10]
)

print(
    probabilities[:3]
)
```

and after convert integer predictions back into their emotion names:

```
predicted_classes = [
    class_names[index]
    for index in predictions
]


print(
    predicted_classes[:10]
)
```



