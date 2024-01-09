"""2.6 概率"""
"""2.6.1 基本概率论"""
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6   # 模拟投掷骰子
print(multinomial.Multinomial(1, fair_probs).sample())  # 1次结果
counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts/1000)  # 得到点数频率

# 图解概率如何随着时间的推移收敛到真实概率
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
print(counts)
cum_counts = counts.cumsum(dim=0) # dim=0 是按照列对每个位置加和
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)   # 广播机制

# print(cum_counts)
# print(cum_counts.sum(dim=1, keepdims=True) )
# print(estimates)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()

