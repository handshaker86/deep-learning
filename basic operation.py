import torch
x = torch.arange(12)
print(x)
print(x.shape)
y=x.reshape(-1,4)  # use -1 can automatically calculate the rest parameter
print(y)
print(torch.ones(2,4,3)) 
z = torch.randn(2,3)
print(z)  # random numbers from normal attribution(average=0,standard deviation=1)
print(torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))   # similar to operations in matlab
print(torch.exp(x))
X = torch.arange(12, dtype=torch.float32).reshape((3,4))   # data type float
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))   # use cat to combine two tensor , dim=0 represent line , dim=1 represent column
print(torch.cat((X, Y), dim=1))
print(X==Y) # logical mask
print(X[:,0:2])  # column 0 and 1 
u = torch.arange(24).reshape(2,3,4)
print(len(u))
print(u.shape)
