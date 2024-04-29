import time
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import torch.nn.functional as F
from torchmetrics.functional import average_precision

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data, 0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1) 
            m.bias.data.zeros_()

def ZscoreNormalization(y):
    for i in range(len(y.columns)):
        indice = y.columns[i]
        y[indice] = (y[indice] - np.mean(y[indice])) / np.std(y[indice])
    return y




def plot_confusion_matrix(Y_true,Y_pred):
  conf_matrix = confusion_matrix(Y_true,Y_pred)
  plt.imshow(conf_matrix, cmap=plt.cm.Greens)
  indices = range(conf_matrix.shape[0])
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


# 定义ROC绘图函数
def ROC_class(preds,Y_test):
    fpr, tpr, threshold = roc_curve(Y_test, preds,pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("ROC:{:.4f}".format(roc_auc))


# 自定义数据集、特征、标签
class MyDataset(Dataset):  # 抽象的类

    def __init__(self, data, label):
        self.data = data
        self.label = label

    # 定义子类
    def __getitem__(self, index):  # 获取样本对
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)


class DNN(nn.Module):
    def __init__(self,input_dims):
        super(DNN,self).__init__()       
        self.fc1=nn.Linear(input_dims,4)
        self.fc2=nn.Linear(4,2)
        self.fc3=nn.Linear(2,2)
        
        self.dropout=nn.Dropout(p=0.4)

        
    def forward(self,x):
        x=x.view(x.shape[0],-1)
        x=self.dropout(F.relu(self.fc1(x)))
        x=self.dropout(F.relu(self.fc2(x)))
            
        # x_out=F.softmax(self.fc3(x),1)
        x_out=self.fc3(x)
        return x_out


f = open("***.csv",encoding='UTF-8')
df = pd.read_csv(f)
label=df.iloc[:,1]
data=df.iloc[:,3:]
input_dim = data.shape[1]
ZscoreNormalization(data)

lr = 0.001
random_seed = 51         #    42, 29,37,49,51,
train_batch_size = 16
val_batch_size = 16
num_epoches = 50
print(input_dim)
print(data.shape)
print(label)
print(data.head())
# print(label)


data_array=np.array(data)
data_array=data_array.astype(np.float32)
label=label.astype(np.float32)

data_tensor=torch.tensor(data_array)
label_tensor=torch.tensor(label)


X_train,X_temp,y_train,y_temp = train_test_split(data_tensor,label_tensor,train_size=0.8,test_size=0.2,random_state=random_seed)
X_test,X_val,y_test,y_val=train_test_split(X_temp,y_temp,test_size=0.5,random_state=random_seed)

trainset = MyDataset(X_train,y_train)
testset = MyDataset(X_test,y_test)
valset = MyDataset(X_val,y_val)
#数据加载器
train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)#原batch_size50
test_loader = DataLoader(testset, batch_size=val_batch_size, shuffle=False)
val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=False)


#设置五个列表来保持指标acc,pre,rec,f1,auc
n1, n2, n3, n4 = [], [], [], []
ACC, PRE, REC, F1 =0, 0, 0, 0
acc_list_train, acc_list_test = [], []
pre_list_test, rec_list_test, f1_list_test = [], [], []
acc_list_val = []
# 记录损失、精确度
train_loss_list, test_loss_list, val_loss_list= [],[],[]
#记录预测结果
y_true_list, y_pred_list = [],[]

model = DNN(input_dim)
initialize_weights(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


#运行时间记录：
"time.clock()默认单位为s,"
start = time.process_time()    #开始记录时间
for epoch in range(num_epoches):
    train_loss, val_loss, val_acc = 0, 0, 0
    model.train()
    out_temp_train, label_temp_train = [],[]
    train_acc = 0
    for data_tensor, label_tensor in train_loader:
        "模型训练阶段，记录loss和acc"
        optimizer.zero_grad()
        outputs = model(data_tensor)
        loss = criterion(outputs,label_tensor.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        outputs_rec_train = F.softmax(outputs,1).argmax(dim=1)
        out_temp_train.extend(outputs_rec_train.detach().numpy())
        label_temp_train.extend(label_tensor.detach().numpy())

        train_acc += accuracy_score(label_temp_train,out_temp_train)

    acc_list_train.append(train_acc / len(train_loader))
    train_loss_list.append(train_loss/len(train_loader))


    model.eval()
    out_temp,label_temp = [],[]

    with torch.no_grad():
        for data_tensor, label_tensor in val_loader:
            "模型验证阶段，记录loss和acc"
            #data_tensor = data_tensor.float()
            outputs = model(data_tensor)
            loss = criterion(outputs,label_tensor.long())
            val_loss +=loss.item()

            outputs_rec = F.softmax(outputs,1).argmax(dim=1)
            out_temp.extend(outputs_rec.detach().numpy())
            label_temp.extend(label_tensor.detach().numpy())
            val_acc +=accuracy_score(label_temp,out_temp)

        val_loss_list.append(val_loss / len(val_loader))
        acc_list_val.append(val_acc / len(val_loader))
     
        if epoch % 2 == 0:
            print("epoch:{},train_loss:{:.4f},val_loss:{:.4f},test_acc:{:.4f}val_acc:{:.4f}"
                .format(epoch,train_loss/len(train_loader),
                    val_loss/len(val_loader),train_acc/len(train_loader),val_acc/len(val_loader)))

end = time.process_time()  #停止时间记录

        
out_temp_test,label_temp_test,out_roc_temp,out_apc_temp = [],[],[],[]
test_acc,test_pre,test_rec,test_f1 = 0,0,0,0
test_loss = 0

for data_tensor, label_tensor in test_loader:
    "测试阶段，评估f1,recall,pre,acc"
    outputs = model(data_tensor)
    loss = criterion(outputs,label_tensor.long())
    test_loss +=loss.item()

    outputs_rec = F.softmax(outputs,1).argmax(dim=1)
    out_temp_test.extend(outputs_rec.detach().numpy())
    label_temp_test.extend(label_tensor.detach().numpy())
    y_true_list.extend(label_tensor.detach().numpy())
    y_pred_list.extend(outputs_rec.detach().numpy())

    out_roc = F.softmax(outputs,1)[:,1]
    out_roc_temp.extend(out_roc.detach().numpy())
    out_apc_temp.extend(out_roc.detach().numpy())

    test_acc +=accuracy_score(label_temp_test,out_temp_test)
    test_pre += precision_score(label_temp_test,out_temp_test)
    test_rec += recall_score(label_temp_test,out_temp_test)
    test_f1 += f1_score(label_temp_test,out_temp_test)

test_loss_list.append(test_loss / len(test_loader))
acc_list_test.append(test_acc / len(test_loader))
pre_list_test.append(test_pre / len(test_loader))
rec_list_test.append(test_rec / len(test_loader))
f1_list_test.append(test_f1 / len(test_loader))

"运行时间计算"
runTime = end - start

print("lr:{:.4f}\nbatch_size:{:.1f}".format(lr,train_batch_size))
print("TEST:acc:{:.4f}\npre:{:.4f}\nrec:{:.4f}\nf1:{:.4f}".format(np.mean(acc_list_test), np.mean(pre_list_test), np.mean(rec_list_test), np.mean(f1_list_test)))
ROC_class(out_roc_temp,label_temp_test)
print("AUPR:{:.4f}".format(average_precision(torch.tensor(out_roc_temp),torch.tensor(label_temp_test).long(),task = "binary")))
# plot_confusion_matrix(y_true_list,y_pred_list)
print("Runtime:{:.3f} Sec".format(runTime))


