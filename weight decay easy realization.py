"""4.5.3 简洁实现"""
# 在下面的代码中，我们在实例化优化器时直接通过weight_decay指定weight decay超参数。
# 默认情况下， PyTorch同时衰减权重和偏移。这里我们只为权重设置了weight_decay，所以偏置参数b不会衰减