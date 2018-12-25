# machine_learning_model_with_api
A simple example of python api for machine learning, using flask.
***

## Files:
* cancer_short.csv - Data for training our model.
                   This file has 11 columns. Please read cancer_readme for more info
* cancer_model.py - Python file for training random forest classifier.
                  uses joblib to dump/create model.pickle file.
* model.pkl - Serialized classification object
* main.py - On running main.py, it loads model.pkl and creates api for predict method
          POST method accespts and returns json.
***
## Running
* Run cancer_model.py to create model.pkl file.
* Run main.py to host locally.
* Request body = [[2,3,1,4,5,2,3,5,4]]
  - An array containg 9 features values ranging between 1-10.
* Responce - ['Prediction : 2']
  - 2 for benign, 4 for malignant
***
## Project in progress.
More to come in future
          
