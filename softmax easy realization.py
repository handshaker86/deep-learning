import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""3.7.1 初始化模型参数"""

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)   # 以均值0和标准差0.01随机初始化权重

net.apply(init_weights)

"""3.7.2 重新审视Softmax的实现"""

loss = nn.CrossEntropyLoss(reduction='none')
#我们没有将softmax概率传递到损失函数中，而是在交叉熵损失函数中传递未规范化的预测，并同时计算softmax及其对数，这是一种类似“LogSumExp技巧” 57的聪明方式。

"""3.7.3 优化算法"""

trainer = torch.optim.SGD(net.parameters(), lr=0.1) # 还是采用随机梯度下降的方法
# 我们使用学习率为0.1的小批量随机梯度下降作为优化算法。这与我们在线性回归例子中的相同，这说明了优化器的普适性。

"""3.7.4 训练"""

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer) # 调用 3.6节中定义的训练函数来训练模型()
d2l.plt.show()

