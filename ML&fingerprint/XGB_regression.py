import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import scipy
from sklearn import metrics
from scipy.stats import pearsonr,spearmanr
from scipy.stats import pearsonr
import time

f=open("***.csv")

df=pd.read_csv(f)

X=df.iloc[:,0:1024]
y=df.iloc[:,1024:]
def ZscoreNormalization(y):
     y=(y-np.mean(y))/np.std(y)
     return y
y=ZscoreNormalization(y)

X_train,X_test_verify,y_train,y_test_verify = train_test_split (
    X, y, random_state=51, train_size = 0.8)
X_test,X_verify,y_test,y_verify = train_test_split (
    X_test_verify,y_test_verify, random_state=51, train_size = 0.5)

A=[]
B=[]
D=[]
Z=[]
varr = 0
for max_depth in range(3,9,1):
    for n_estimators in range(100,1100,10):
        xgb_model=xgb.XGBRegressor(max_depth=max_depth,
                                       n_estimators=n_estimators,
                                       learning_rate=0.05,
                                       objective="reg:squarederror",
                                       booster="gbtree",
                                       )
        xgb_model.fit(X_train,y_train)
        y_predic=xgb_model.predict(X_verify)
        MSE=mean_squared_error(y_verify,y_predic)
        RMSE=np.sqrt(MSE)
        A.append(max_depth)
        B.append(n_estimators)
        D.append(RMSE)
        varr+=1

location=D.index(min(D))  # 返回最小值
print("最优n_estimators是"+str(B[location]))
print("验证集xgboost回归最优rmse是"+str(D[location]))
bestestimator=B[location]
bestmaxdepth=A[location]

xgb_modelfinal= xgb.XGBRegressor(max_depth=bestmaxdepth,
                              n_estimators=bestestimator,  # 0.419,
                              learning_rate=0.05,  # 400,
                              objective="reg:squarederror",
                              booster="gbtree",
                              )
start=time.time()
xgb_modelfinal.fit(X_train,y_train)
end=time.time()
y_predicfinal= xgb_modelfinal.predict(X_test)
MSEF=mean_squared_error(y_test,y_predicfinal)

RMSEF=np.sqrt(MSEF)
MAE = mean_absolute_error(y_test,y_predicfinal)
R2 = r2_score(y_test,y_predicfinal)
pcc=pearsonr(y_test.squeeze(), y_predicfinal.squeeze())[0]
spea=spearmanr(y_test.squeeze(), y_predicfinal.squeeze())[0]


Z.append(RMSEF)
locationfinal=Z.index(min(Z))
print("测试集xgboost回归最优rmse是"+str(Z[locationfinal]))
print("MAE:",MAE)
print("PCC:",pcc)
print("R2:",R2)
print("SP:",spea)
print("XGBoost程序最优参数训练时间为：{}".format(end-start))

