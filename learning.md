
## 二、chp2

### 1. np.outer, 外积计算

```python
expected(i, j) = col_totals[i] * row_totals[j] # 计算pmi的时候可以用到
```

### 2. svd

- 特征值与特征向量:https://www.bilibili.com/video/BV1vY4y1J7gd?spm_id_from=333.337.search-card.all.click&vd_source=953ee6e625bf100ef80ae35363880522
```python
A*v1 = l*v1: 这里矩阵只起到了伸缩的作用，l就是特征值，v1就是特征向量。

```
- svd介绍:https://www.bilibili.com/video/BV1Hq4y117VK?spm_id_from=333.337.search-card.all.click&vd_source=953ee6e625bf100ef80ae35363880522
```python
W(m*n) = U(旋转，m*m)*s(拉伸, m*n)*Vh(旋转, n*n)
```

- svd用法，降低纬度:https://www.cnblogs.com/pinard/p/6251584.html
```
对于奇异值,它跟我们特征分解中的特征值类似，在奇异值矩阵中也是按照从大到小排列，而且奇异值的减少特别的快，在很多情况下，前10%甚至1%的奇异值的和就占了全部的奇异值之和的99%以上的比例。也就是说，我们也可以用最大的k个的奇异值和对应的左右奇异向量来近似描述矩阵。也就是说其中k要比n小很多，也就是一个大的矩阵A可以用三个小的矩阵𝑈𝑚×𝑘,Σ𝑘×𝑘,𝑉𝑇𝑘×𝑛来表示。如下图所示，现在我们的矩阵A只需要灰色的部分的三个小矩阵就可以近似描述了。
```

## 二、chp3

### 序列标注

- crf

- 维特比算法（Viterbi）解码算法

- 语法树解析

- 依存句法分析

- hmm:【数之道 29】"隐马尔可夫模型"HMM是什么？了解它只需5分钟！https://www.bilibili.com/video/BV1ZB4y1y7gC?spm_id_from=333.337.search-card.all.click&vd_source=953ee6e625bf100ef80ae35363880522



## 四、基础代码学习

### 4.1 mlp
```python
import torch
from torch import nn, optim
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs):
        hidden = self.linear1(inputs)
        activation = self.activate(hidden)
        outputs = self.linear2(activation)
        # 获得每个输入属于某一类别的概率（Softmax），然后再取对数
        # 取对数的目的是避免计算Softmax时可能产生的数值溢出问题
        # log_softmax + NLLLoss() 就是交叉熵的结果
        # Pytorch详解NLLLoss和CrossEntropyLoss : https://blog.51cto.com/u_15274944/2921745
        # 损失函数｜交叉熵损失函数 ：https://zhuanlan.zhihu.com/p/35709485
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs

# 异或问题的4个输入
x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
# 每个输入对应的输出类别
y_train = torch.tensor([0, 1, 1, 0])

# 创建多层感知器模型，输入层大小为2，隐含层大小为5，输出层大小为2（即有两个类别）
model = MLP(input_dim=2, hidden_dim=5, num_class=2)

criterion = nn.NLLLoss() # 当使用log_softmax输出时，需要调用负对数似然损失（Negative Log Likelihood，NLL）
optimizer = optim.SGD(model.parameters(), lr=0.05) # 使用梯度下降参数优化方法，学习率设置为0.05

for epoch in range(100):
    y_pred = model(x_train) # 调用模型，预测输出结果
    loss = criterion(y_pred, y_train) # 通过对比预测结果与正确的结果，计算损失
    optimizer.zero_grad() # 在调用反向传播算法之前，将优化器的梯度值置为零，否则每次循环的梯度将进行累加
    loss.backward() # 通过反向传播计算参数的梯度
    optimizer.step() # 在优化器中更新参数，不同优化器更新的方法不同，但是调用方式相同

print("Parameters:")
for name, param in model.named_parameters():
    print (name, param.data)

y_pred = model(x_train)
print("Predicted results:", y_pred.argmax(axis=1))

```

### 4.2 embedding
```python
import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 线性变换：词嵌入层->隐含层
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        # 使用ReLU激活函数
        self.activate = F.relu
        # 线性变换：激活层->输出层
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs):
        # B*S = 2 * 4
        embeddings = self.embedding(inputs)
        # 将序列中多个embedding进行聚合（此处是求平均值）
        embedding = embeddings.mean(dim=1)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        # 获得每个序列属于某一类别概率的对数值
        probs = F.log_softmax(outputs, dim=1)
        return probs

mlp = MLP(vocab_size=8, embedding_dim=3, hidden_dim=5, num_class=2)
# 输入为两个长度为4的整数序列
inputs = torch.tensor([[0, 1, 2, 1], [4, 6, 6, 7]], dtype=torch.long)
outputs = mlp(inputs)
print(outputs)
```