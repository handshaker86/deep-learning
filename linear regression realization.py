"""3.2 线性回归的从零开始实现"""
import random
import torch
from d2l import torch as d2l

"""3.2.1 生成数据集"""

def synthetic_data(w, b, num_examples): #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 补上噪声项
    return X, y.reshape((-1, 1))   # 产生人工数据集

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) # 调用函数
d2l.set_figsize()
d2l.plt.scatter(features[:, (0)].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()  # 通过散点图可观察到features与lable之间的线性关系

"""3.2.2 读取数据集"""

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])   # 卡住区间长度为batch_size
        yield features[batch_indices], labels[batch_indices]
# 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量,每个小批量包含一组特征和标签。
# yield 类似于return，yield返回一个可迭代的 generator（生成器）对象，你可以使用for循环或者调用next()方法遍历生成器对象

"""3.2.3 初始化模型参数"""

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)  
# 从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，并将偏置初始化为0。
# 每次更新都需要计算损失函数关于模型参数的梯度。有了这个梯度，我们就可以向减小损失的方向更新每个参数

"""3.2.4 定义模型"""

def linreg(X, w, b): #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b   # matmul就是将X和w点乘

"""3.2.5 定义损失函数"""

def squared_loss(y_hat, y): #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
# 需要将真实值y的形状转换为和预测值y_hat的形状相同

"""3.2.6 定义优化算法"""
def sgd(params, lr, batch_size): #@save
    """小批量随机梯度下降"""
    with torch.no_grad():    # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
        for param in params:
            param -= lr * param.grad / batch_size   # 此处lr表示学习率
            param.grad.zero_()

"""3.2.7 训练"""

lr = 0.03
num_epochs = 3   # 设置超参数
batch_size = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。 l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
