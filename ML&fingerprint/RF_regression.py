f=open("***.csv")
df=pd.read_csv(f)

label=df.iloc[:,2]
data=df.iloc[:,5:]
print(label.head())
print(data.head())
print(data.shape[1])

X_train,X_temp,y_train,y_temp = train_test_split(data,label,train_size=0.8,test_size=0.2,random_state=42)
X_test,X_val,y_test,y_val=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42)
#################################################################################################################
rmse_min = 10
# n_estimators_best, max_depth_best = 400, 30
random_seed=42
random_forest_seed=np.random.randint(low=1,high=230)


for e in range(50,400,50):
  for md in range(20,40,5):
    model = RandomForestRegressor(n_estimators=e, max_depth=md, random_state=random_forest_seed)
    model.fit(X_train,y_train)
    y_preds = model.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val,y_preds))
    print('rmse：',rmse,'n_estimators:',e,'max_depth:',md)
    if rmse < rmse_min:
        rmse_min=rmse
        n_estimators_best=e
        max_depth_best=md
        mae = mean_absolute_error(y_val, y_preds)
        pcc = pearsonr(y_val,y_preds)
        r2 = r2_score(y_val,y_preds)
print('val:rmse:{:.3f}'.format(rmse_min))
print('val:mae:{:.3f}'.format(mae))
print('val:r2:{:.3f}'.format(r2))
print('val:PCC:',pcc)
print('n_estimators_best:',n_estimators_best,'max_depth_best:',max_depth_best)



predictor =RandomForestRegressor(n_estimators=n_estimators_best,max_depth=max_depth_best,random_state=random_forest_seed)


start = time.process_time()    #开始记录时间
predictor.fit(X_train, y_train)
end = time.process_time()  #停止时间记录

y_pred = predictor.predict(X_test)
y_pred = np.array(y_pred)
y_test = np.array(y_test)

RMSE = sqrt(mean_squared_error(y_test,y_pred))
MAE = mean_absolute_error(y_test, y_pred)
PCC = pearsonr(y_test,y_pred)
R2 = r2_score(y_test,y_pred)
SP = spearmanr(y_test,y_pred)

"运行时间计算"
runTime = end - start

print("RMSE:{:.3f}".format(RMSE))
print("MAE:{:.3f}".format(MAE))
print("PCC:",PCC[0])
print("R2:{:.3f}".format(R2))
print("Spear:",SP[0])
print("Runtime:{:.3f} Sec".format(runTime))


