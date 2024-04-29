import pandas as pd
import numpy as np
import time
from math import sqrt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import skmultilearn
from skmultilearn.problem_transform import BinaryRelevance
# import pydot

import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import metrics
# from openpyxl import load_workbook
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, auc, roc_curve, roc_auc_score, precision_recall_curve 
from sklearn.metrics import average_precision_score, precision_score,recall_score,accuracy_score

# 定义ROC绘图函数
def ROC_class(Y_test,preds):
    fpr, tpr, threshold = roc_curve(Y_test, preds,pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("auc={:.2f}".format(roc_auc))

    plt.figure(figsize=(6,6))
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


    
# 定义PRC绘图函数
def PR_class(model, X_test, y_test):
    preds = model.predict(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, preds, pos_label=1)
    prc_auc = average_precision_score(y_test, preds, average='macro', pos_label=1, sample_weight=None)

    plt.figure()
    plt.step(recall, precision, label=' (PRC={:.4f})'.format(prc_auc))
    plt.plot([0, 1], [1, 0], 'r--')
    plt.title('PR curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.show()

    print('AUPRC：', prc_auc)
      
def plot_confusion_matrix(Y_true,Y_pred):
  conf_matrix = confusion_matrix(Y_true,Y_pred)
  plt.imshow(conf_matrix, cmap=plt.cm.Greens)
  indices = range(conf_matrix.shape[0])
  #labels = [0,1,2,3,4,5,6,7,8,9]
  #plt.xticks(indices, labels)
  #plt.yticks(indices, labels)
  plt.xticks(indices)
  plt.yticks(indices)
  plt.colorbar()
  plt.xlabel('y_pred')
  plt.ylabel('y_true')
  # 显示数据
  for first_index in range(conf_matrix.shape[0]):
    for second_index in range(conf_matrix.shape[1]):
      plt.text(first_index, second_index, conf_matrix[first_index, second_index])
  #plt.savefig('heatmap_confusion_matrix.jpg')
  plt.ylabel('Real label')
  plt.xlabel('Prediction')
  plt.show()

f=open("***.csv", encoding='UTF-8')
df=pd.read_csv(f)

df.head()
label=df.iloc[:,2]
data=df.iloc[:,597:]
print(label.head())
print(data.head())
print(data.shape)



X_train,X_temp,y_train,y_temp = train_test_split(data,label,train_size=0.8,test_size=0.2,random_state=42)
X_test,X_val,y_test,y_val=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42)
#################################################################################################################
f1 = 0.01
n_estimators_best, max_depth_best = 400, 30
random_seed=42
random_forest_seed=np.random.randint(low=1,high=230)


for e in range(50,400,50):
  for md in range(10,40,5):
    # '单标签分类方法：单分类'
    model = RandomForestClassifier(n_estimators=e, max_depth=md, random_state=random_forest_seed)

    model.fit(X_train,y_train)
    y_preds = model.predict(X_val)
    f = f1_score(y_val,y_preds,average='binary')
    print('VAL_f1：',f,'n_estimators:',e,'max_depth:',md)
    if f > f1:
        f1=f
        n_estimators_best=e
        max_depth_best=md
print('VAL_f1：',f1)
print('n_estimators_best:',n_estimators_best,'max_depth_best:',max_depth_best)



predictor = RandomForestClassifier(n_estimators=e, max_depth=md, random_state=random_forest_seed)

#运行时间记录：
"time.clock()默认单位为s,"
start = time.process_time()    #开始记录时间
predictor.fit(X_train, y_train)
end = time.process_time()  #停止时间记录

y_pred = predictor.predict(X_test)
y_pred = np.array(y_pred.astype(int))
y_test = np.array(y_test.astype(int))
y_pred_proba = predictor.predict_proba(X_test)[:,1]


precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
f1score = f1_score(y_test,y_pred)
"运行时间计算"
runTime = end - start

print("Precision={:.2f}".format(precision))
print("Recall={:.2f}".format(recall))
print("Accuracy={:.2f}".format(accuracy))
print("F1_Score={:.2f}".format(f1score))
ROC_class(y_test,y_pred_proba)
plot_confusion_matrix(y_test,y_pred)
print("Runtime:{:.3f} Sec".format(runTime))




