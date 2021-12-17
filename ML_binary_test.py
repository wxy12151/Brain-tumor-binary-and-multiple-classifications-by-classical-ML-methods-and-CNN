import glob
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, roc_curve, auc

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

# binary classification
labels_csv = pd.read_csv('AMLS-2021_test/test/label.csv')
Y_2_test = labels_csv['label']
Y_2_test[Y_2_test != 'no_tumor'] = 1
Y_2_test[Y_2_test == 'no_tumor'] = 0

Y_2_test = Y_2_test.astype('int')

## SVM
clf_svm = joblib.load('save/clf_svm_binary_final.pkl')
y_pred = clf_svm.predict(X_scaled_test)
print('SVM')
print('Accuracy on test set: '+str(accuracy_score(Y_2_test,y_pred)))
print("Confusion Matrix: ",'\n', confusion_matrix(Y_2_test,y_pred))
print("Classification report: ",'\n',classification_report(Y_2_test,y_pred))

## knn
clf_knn = joblib.load('save/clf_knn_binary_final.pkl')
y_pred = clf_knn.predict(X_scaled_test)
print('KNN')
print('Accuracy on test set: '+str(accuracy_score(Y_2_test,y_pred)))
print("Confusion Matrix: ",'\n', confusion_matrix(Y_2_test,y_pred))
print("Classification report: ",'\n',classification_report(Y_2_test,y_pred))

### RF
clf_RF = joblib.load('save/clf_RF_binary_final.pkl')
y_pred = clf_RF.predict(X_scaled_test)
print('Random Forest')
print('Accuracy on test set: '+str(accuracy_score(Y_2_test,y_pred)))
print("Confusion Matrix: ",'\n', confusion_matrix(Y_2_test,y_pred))
print("Classification report: ",'\n',classification_report(Y_2_test,y_pred))
scores = clf_RF.predict_proba(X_scaled_test)


### ROC for SVM
scores_svm = clf_svm.predict_proba(X_scaled_test)
fpr, tpr, thresholds = roc_curve(Y_2_test, scores_svm[:, 1], pos_label=1) # positive label = 1
auc_ = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of SVM')
plt.legend(loc="lower right")
# plt.savefig('save/roc of SVM.png')
plt.show()

### ROC for KNN
scores_knn = clf_knn.predict_proba(X_scaled_test)
fpr, tpr, thresholds = roc_curve(Y_2_test, scores_knn[:, 1], pos_label=1) # positive label = 1
auc_ = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of KNN')
plt.legend(loc="lower right")
# plt.savefig('save/roc of KNN.png')
plt.show()

### ROC for RF
scores_rf = clf_RF.predict_proba(X_scaled_test)
fpr, tpr, thresholds = roc_curve(Y_2_test, scores_rf[:, 1], pos_label=1) # positive label = 1
auc_ = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of RF')
plt.legend(loc="lower right")
# plt.savefig('save/roc of RF.png')
plt.show()