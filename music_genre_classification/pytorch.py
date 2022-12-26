import numpy as np
import Codes.Layer as Layer
import matplotlib.pyplot as plt
import Codes.optimizer as optimizer
from Data.data_preprocessing import data_preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F

(x_train, t_train), (x_test, t_test) = data_preprocessing()

x_train = torch.Tensor(x_train)
t_train = torch.Tensor(t_train)
x_test = torch.Tensor(x_test)
t_test = torch.Tensor(t_test)

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(x_train.shape[1], 50, bias=True)
        self.linear2 = nn.Linear(50, 100, bias=True)
        self.linear3 = nn.Linear(100, 100, bias=True)
        self.linear4 = nn.Linear(100, 50, bias=True)

        self.linear5 = nn.Linear(50, t_train.shape[1], bias=True)
        self.relu = torch.nn.ReLU()
        self.bn1 = nn.BatchNorm1d(50)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(50)

        
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.linear4.weight)
        self.model = nn.Sequential(self.linear1, self.bn1, self.relu, 
                                   self.linear2, self.bn2, self.relu, 
                                   self.linear3, self.bn3, self.relu,
                                   self.linear4, self.bn4, self.relu,
                                   self.linear5)
    
    def forward(self, x):
        return self.model(x)
net = SoftmaxClassifierModel()
inputs = x_train
print(inputs)
print(net(inputs))
import torch.optim as optim
t_test = torch.max(t_test.data, 1)[1] 
t_ttttt = torch.max(t_train.data, 1)[1] 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_list = []
train_acc = []
test_acc = []

for epoch in range(500):   # 데이터셋을 수차례 반복합니다.
    net.train()

    running_loss = 0.0
    # [inputs, labels]의 목록인 data로부터 입력을 받은 후;

    # 변화도(Gradient) 매개변수를 0으로 만들고
    optimizer.zero_grad()

    # 순전파 + 역전파 + 최적화를 한 후
    outputs = net(x_train)
    loss = criterion(outputs, t_train)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.data)
    if epoch % 10 == 0:
        with torch.no_grad(): # grad 해제 
            net.eval()
            correct = 0
            total = 0
            out = net(x_train)
            preds = torch.max(out.data, 1)[1] # 출력이 분류 각각에 대한 값으로 나타나기 때문에, 가장 높은 값을 갖는 인덱스를 추출
            total += len(t_ttttt) # 전체 클래스 개수 
            correct += (preds==t_ttttt).sum().item() # 예측값과 실제값이 같은지 비교
            train_acc.append(100.*correct/total)    
            print('Train Accuracy: ', 100.*correct/total, '%')
            
            correct = 0
            total = 0
            out = net(x_test)
            preds = torch.max(out.data, 1)[1] # 출력이 분류 각각에 대한 값으로 나타나기 때문에, 가장 높은 값을 갖는 인덱스를 추출
            total += len(t_test) # 전체 클래스 개수 
            correct += (preds==t_test).sum().item() # 예측값과 실제값이 같은지 비교
            test_acc.append(100.*correct/total)    
            print('Test Accuracy: ', 100.*correct/total, '%')


print(loss_list)
print(train_acc)
print(test_acc)

plt.plot(loss_list)
plt.show()