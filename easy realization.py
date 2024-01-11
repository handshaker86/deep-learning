"""3.3 线性回归的简洁实现"""
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn  # nn是神经网络的缩写

"""3.3.1 生成数据集"""
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

"""3.3.2 读取数据集"""
def load_array(data_arrays, batch_size, is_train=True): #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

"""3.3.3 定义模型"""

net = nn.Sequential(nn.Linear(2, 1)) 
# Sequential实例将数据传入到第一层，然后将第一层的输出作为第二层的输入，以此类推
#第一个指定输入特征形状，即2;第二个指定输出特征形状，输出特征形状为单个标量，因此为1。

"""3.3.4 初始化模型参数"""

net[0].weight.data.normal_(0, 0.01)  # net[0]选择第一个图层，初始化w和b
net[0].bias.data.fill_(0)

"""3.3.5 定义损失函数"""

loss = nn.MSELoss()
# 计算均方误差使用的是MSELoss类,默认情况下,它返回所有样本损失的平均值

"""3.3.6 定义优化算法"""

trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# 当我们实例化一个SGD实例时，我们要指定优化的参数（可通过net.parameters()从我们的模型中获得）
# 以及优化算法所需的超参数字典。小批量随机梯度下降只需要设置lr值，这里设置为0.03。

"""3.3.7 训练"""

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()   # 更新w和b
    l = loss(net(features), labels)   # 用更新后的w和b计算loss
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差: ', true_b - b)
