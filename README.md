# Setup

## Environment: 

CPU: Intel 8- Core i7-11800

GPU: RTX 3060

Pytorch 1.10 with cuda11.3 cudnn8.0

## External libraries:

PyTorch, sklearn, pandas, PIL, matplotlib,  tensorboard, joblib, glob

# 1. Brief introduction

The brain tumors are the common and  aggressive disease that can be detected by magnetic resonance imaging (MRI) figures. The purpose of this project is to explore the classification power of different  algorithms based on brain MRI scans.

## 1.1. Binary task - Random Forest, SVM, KNN

Identify whether there is a tumor in MRI images

## 1.2. Multi-class task - MLP, CNN

Classify glioma, meningioma,  pituitary tumors, and no tumor.

# 2. Organization of the files

## 2.1. Dataset

### 2.1.1. Training and validation dataset

#### Folder "AMLS-2021_dataset": 

3000 512x512 pixel gray-scale MRI images organized in 4  classes.

860 glioma images, 855 meningioma images, 831 pituitary images and 454 no tumor images.

### 2.1.2. Testing dataset

#### Folder "AMLS-2021_test": 

200 512x512 pixel gray-scale MRI images organized in 4  classes.

43 glioma images, 68 meningioma images, 52 pituitary images and 37 no tumor images.

## 2.2. Binary task - Random Forest, SVM, KNN

### File "ML_binary_all.ipynb"  is written by Jupyter Notebook which contains the process of training:

1. Load and pre-process the dataset. (test: valid = 8:2)

2. Hyper-parameters selection process.

3. Learning curve.

4. save the model.

### File "ML_binary_test.py" is used for testing the models:

1. Load the models saved before.

2. The same pre-processing as training.

3. Testing the models by accuracy, confusion matrix and classification report.

4. Draw the ROC curves of random forest, SVM and KNN.

## 2.3. Muiti-class task - MLP, CNN

### File "model.py"  save the network structures of MLP and CNN in this project.

### File "DL_train.py" trains and saves the model in "model_save".

### File "DL_test.py" tests the model and output statistical metrics and ROC curves.

# 3. Run the code

## 3.1. Binary task

### Training: run "ML_binary_all.ipynb" in order in Jupyter.

### Testing: run "ML_binary_test.py"

Notes. Need the model files but not provide here.

## 3.2. Multi-class task

### Training: run "DL_train.py".

### Testing: run "DL_test.py"

Notes. Need the model files but only one CNN model is provided here.

### DO NOT CHANGE THE FILE LEVEL

























