import glob
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

images_test = []
for f in glob.iglob("AMLS-2021_test/test/image/*"):
    img = Image.open(f)
    img_resize = img.resize((256, 256))
    images_test.append(np.asarray(img_resize))
### grey image
X_origin_test = np.array(images_test)[:,:,:,0]
### reshape to 256 x 256 (the same as training set)
X = X_origin_test.reshape(X_origin_test.shape[0],-1)#3000x65536 256^2
X_scaled_test = preprocessing.scale(X)
pca = joblib.load('save/pca_0.95.m')
X_pca_test = pca.transform(X_scaled_test)
print(X_pca_test.shape)

# binary classification
labels_csv = pd.read_csv('AMLS-2021_test/test/label.csv')
Y_2_test = labels_csv['label']
Y_2_test[Y_2_test != 'no_tumor'] = 1
Y_2_test[Y_2_test == 'no_tumor'] = 0

Y_2_test = Y_2_test.astype('int')


### SVM
clf_svm = joblib.load('save/clf_svm_binary.pkl')
y_pred = clf_svm.predict(X_scaled_test)
print('Accuracy on test set: '+str(accuracy_score(Y_2_test,y_pred)))
print(classification_report(Y_2_test,y_pred))

### knn
clf_knn = joblib.load('save/clf_knn_binary.pkl')
y_pred = clf_knn.predict(X_scaled_test)
print('Accuracy on test set: '+str(accuracy_score(Y_2_test,y_pred)))
print(classification_report(Y_2_test,y_pred))

### RF
clf_RF = joblib.load('save/clf_RF_binary.pkl')
y_pred = clf_RF.predict(X_scaled_test)
print('Accuracy on test set: '+str(accuracy_score(Y_2_test,y_pred)))
print(classification_report(Y_2_test,y_pred))


