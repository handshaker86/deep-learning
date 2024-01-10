import torch

"""2.5 自动微分"""
"""2.5.1 一个简单的例子"""
x = torch.arange(4.0)
x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)

# 创建一个张量x，并设置其 requires_grad参数为True，程序将会追踪所有对于该张量的操作，当完成计算后通过调用.backward，自动计算所有的梯度， 
# 这个张量的所有梯度将会自动积累到x的.grad属性

x.grad # 默认值是None
y = 2 * torch.dot(x, x)
print(y)
y.backward()   # 假设x=[x1,x2]，y就等于2*（x1^2+x2^2),y.backward即对y分别求x1,x2的偏导，将两个结果作为一个向量输出在x.grad中
print(x.grad)  # 打印这些梯度

# 在默认情况下， PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

"""2.5.2 非标量变量的反向传播"""
x.grad.zero_()
y = x * x           # 调用backward的变量必须是标量，否则要在backward（）括号中传入torch.ones(len(x))，作为最终grad输出的系数
                    # 对非标量调用backward需要传入一个gradient参数（即torch.ones(len(x)
y.sum().backward()  # 等价于y.backward(torch.ones(len(x)))
print(x.grad)

x.grad.zero_()
y = x * x   
y.backward(torch.ones(len(x)))
print(x.grad)

"""2.5.3 分离计算"""
x.grad.zero_()
y = x * x
u = y.detach()
# detach()方法用于返回一个新的 Tensor，这个 Tensor 和原来的 Tensor 共享相同的内存空间
# 但是这个Tensor不会被计算图所追踪，也就是说它不会参与反向传播，不会影响到原有的计算图
z = u * x
z.sum().backward()
print(x.grad == u) # 在调用backward时，u会被看做一个常数

"""2.5.4 Python控制流的梯度计算"""
def f(a):
    b = a * 2
    while b.norm() < 1000:   # 默认是2范数（即sqrt（x1^2+x2^2……））
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)  # 来自正态分布的随机数(average=0,standard deviation=1)
d = f(a)  # 由f（a）看出，d = k * a , 可通过下面验证
d.backward()
print(a.grad == d / a)  
