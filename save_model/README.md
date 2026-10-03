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

