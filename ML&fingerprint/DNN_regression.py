from cmath import sqrt
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchmetrics
from torchmetrics.functional import mean_squared_error, mean_absolute_error, pearson_corrcoef,r2_score,spearman_corrcoef
from sklearn import model_selection

from scipy.stats import pearsonr
from tqdm import tqdm


class MyDataset(Dataset):  # 抽象的类

    def __init__(self, data, label):
        self.data = data
        self.label = label

    # 定义子类
    def __getitem__(self, index):  # 获取样本对
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

#定义多维特征的PCC函数
def PCC_score(y_true, y_pred):
    n_tasks = len(y_true[1])
    scores = []

    for task in range(n_tasks):
        task_y_true = y_true[:, task]
        task_y_pred = y_pred[:, task]

        task_score = pearson_corrcoef(task_y_true, task_y_pred)

        if not torch.isnan(task_score):
            scores.append(task_score.detach().numpy())
    result = np.mean(scores)
    return result

#定义多维特征的spear函数
def SP_score(y_true, y_pred):
    n_tasks = len(y_true[1])
    scores = []

    for task in range(n_tasks):
        task_y_true = y_true[:, task]
        task_y_pred = y_pred[:, task]

        task_score = spearman_corrcoef(task_y_true, task_y_pred)

        if not torch.isnan(task_score):
            scores.append(task_score.detach().numpy())
    result = np.mean(scores)
    return result



class DNN(nn.Module):
    def __init__(self,input_dims,output_dims):
        super(DNN,self).__init__()       
        self.fc1=nn.Linear(input_dims,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,output_dims)
        
        self.dropout=nn.Dropout(p=0.2)

        
    def forward(self,x):
        x=x.view(x.shape[0],-1)
        x1=self.dropout(F.relu(self.fc1(x)))
        x2=self.dropout(F.relu(self.fc2(x1))) 
        # x_out=F.softmax(self.fc3(x),1)
        x_out=self.fc3(x2)
        return x_out


# 标签数据标准化
def ZscoreNormalization(y):
    y = (y - np.mean(y)) / np.std(y)
    return y


def commonTyoe(tensor):
    if torch.cuda.is_available():
        cuda = "cuda:0"
        return tensor.cuda(cuda)
    else:
        return tensor


# 处理数据
f = open("***.csv")
df=pd.read_csv(f)


label=df.iloc[:,9]
data=df.iloc[:,11:]
label = ZscoreNormalization(label)

lr = 0.0001
random_seed = 51             #     29,37,49,51,42
train_batch_size = 32
val_batch_size = 32
num_epoches = 100


print(label.head())
print(data.head())
print(data.shape[1])


input_dim = data.shape[1]
output_dim = 1
print('input_dim:\t',input_dim)
print('output_dim:\t',output_dim)





data_array=np.array(data)
data_array=data_array.astype(np.float32)
label=np.array(label)
label=label.astype(np.float32)

data_tensor=torch.tensor(data_array)
label_tensor=torch.tensor(label)


model = DNN(input_dim,output_dim)
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
rmse_list, mae_list, r2_score_list, pcc_max_list, pcc_ave_list = [], [], [], [], []

# for train_index, test_index in kf.split(df):
    # 记录损失、精确度
train_loss_list = []
train_r2_list = []
train_rmse_list = []

test_loss_list = []
test_r2_list = []
test_rmse_list = []
test_mae_list = []
test_pcc_list = []
test_sp_list = []

val_loss_list = []
val_r2_list = []
val_rmse_list = []



X_train,X_temp,y_train,y_temp = train_test_split(data_tensor,label_tensor,train_size=0.8,test_size=0.2,random_state=42)
X_test,X_val,y_test,y_val=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42)


trainset = MyDataset(X_train,y_train)
testset = MyDataset(X_test,y_test)
valset = MyDataset(X_val,y_val)
#数据加载器
train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)#原batch_size50
test_loader = DataLoader(testset, batch_size=val_batch_size, shuffle=False)
val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=False)




rmse, mae, r2, = 0, 0, 0
pcc_max, pcc_ave = 0, 0
n1, n2, n3, n4 = [], [], [], []
batch_r2score = torchmetrics.R2Score()


"time.clock()默认单位为s,"
start = time.process_time()    #开始记录时间

for epoch in range(num_epoches):      #在迭代体上添加tqdm模块，来美化训练过程，呈现训练进度条
    model.train()
    train_loss = 0
    train_r2 = 0
    for data_tensor, label_tensor in train_loader:
        optimizer.zero_grad()
        # 输出
        output = model(data_tensor)
        output1 = output.squeeze(-1)  #squeeze(-1)的作用是将列向量变为行向量
        # 计算损失
        loss = criterion(output1, label_tensor)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()


        train_loss += loss.item()
        train_r2 += r2_score(output1, label_tensor)

    train_loss_list.append(train_loss / len(train_loader))
    train_r2_list.append(train_r2 / len((train_loader)))

    # 在测试集上运行 每一次epoch都要将一下清零
    model.eval()

    val_loss = 0
    val_r2 = 0


    for data_tensor, label_tensor in val_loader:
        if len(data_tensor) == 1:
            continue
        output = model(data_tensor)
        output1 = output.squeeze(-1)
        loss = criterion(output1, label_tensor)
        val_loss += loss.item()
        batch_r2score = torchmetrics.R2Score()
        # test_r2 += batch_r2score(output_1, label_tensor)


        val_r2 += r2_score(output1, label_tensor)  # _1

    val_loss_list.append(val_loss / len(val_loader))
    val_r2_list.append(val_r2 / len(val_loader))

    if epoch % 2 == 0:
        print("epoch:{},train_loss:{:.4f},val_loss:{:.4f},train_r2:{:.4f}val_r2:{:.4f}"
            .format(epoch,train_loss/len(train_loader),
                val_loss/len(val_loader),train_r2/len(train_loader),val_r2/len(val_loader)))

end = time.process_time()  #停止时间记录
"运行时间计算"
runTime = end - start

##########################################################################################################
test_loss, test_r2, test_rmse, test_mae, pcc_ave, sp_ave= 0,0,0,0,0,0

for data_tensor, label_tensor in tqdm(test_loader) :
    output = model(data_tensor)
    output1 = output.squeeze(-1)
    loss = criterion(output1, label_tensor)
    test_loss += loss.item()
    batch_r2score = torchmetrics.R2Score()
    # test_r2 += batch_r2score(output_1, label_tensor)


    test_r2 += r2_score(output1, label_tensor)  # _1
    test_rmse += sqrt(mean_squared_error(output1, label_tensor))
    test_mae += mean_absolute_error(output1, label_tensor)

    #pcc
    "存在问题：pearson_corrcoef函数只接受输入特征为1维的向量，而output1和label_tensor为非一维特征的矩阵"
    "问题已解决"
    test_pcc = pearson_corrcoef(output1, label_tensor) 
    pcc_ave += test_pcc

    #spearman
    test_sp = spearman_corrcoef(output1, label_tensor) 
    sp_ave += test_sp


test_loss_list.append(test_loss / len(test_loader))
test_r2_list.append(test_r2 / len(test_loader))
test_mae_list.append(test_mae / len(test_loader))
test_rmse_list.append(test_rmse / len(test_loader))
test_pcc_list.append(pcc_ave / len(test_loader))
test_sp_list.append(sp_ave / len(test_loader))



print("lr:{:.4f}\nbatch_size:{:.1f}".format(lr,train_batch_size))
print("RMSE:{:.4f}\nMAE:{:.4f}\nR2_SCORE:{:.4f}\nPCC:{:.4f}\nSpear:{:.4f}".format(test_rmse / len(test_loader),
                                                        test_mae / len(test_loader),
                                                        test_r2 / len(test_loader),
                                                        pcc_ave / len(test_loader),
                                                        sp_ave / len(test_loader)
                                                        ))

print("Runtime:{:.3f} Sec".format(runTime))


