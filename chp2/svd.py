# Defined in Section 2.1.2

import numpy as np
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

M = np.array([[0, 2, 1, 1, 1, 1, 1, 2, 1, 3],
              [2, 0, 1, 1, 1, 0, 0, 1, 1, 2],
              [1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
              [1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
              [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
              [1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
              [2, 1, 0, 0, 0, 1, 1, 0, 1, 2],
              [1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
              [3, 2, 1, 1, 1, 1, 1, 2, 1, 0]])

def pmi(M, positive=True):
    """求pmi
    Args:
        M(n*n): 共现矩阵，n为词汇的数量，M(i,j)代表词汇i和词汇j共现次数。
    Returns:
        pmi(n*n): p(i,j)代表词汇i和词汇j的紧密度, pmi = log2[p(w, c)/(p(w) * p(c))]。
    """
    # 每一列的累积和
    col_totals = M.sum(axis=0)
    # 每一行的累积和，其实和每一列的相等
    row_totals = M.sum(axis=1)
    total = col_totals.sum()
    # 外积，expected(i, j) = col_totals[i] * row_totals[j]
    expected = np.outer(row_totals, col_totals) / total
    M = M / expected # p(w, c)/(p(w) * p(c))
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        M = np.log(M)
    M[np.isinf(M)] = 0.0  # log(0) = 0
    if positive:
        M[M < 0] = 0.0
    return M

M_pmi = pmi(M)

np.set_printoptions(precision=2)
print(M_pmi)

# b避免稀疏性，反应高阶共现关系
U, s, Vh = np.linalg.svd(M_pmi)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

words = ["我", "喜欢", "自然", "语言", "处理", "爱", "深度", "学习", "机器", "。"]

for i in range(len(words)):
    plt.text(U[i, 0], U[i, 1], words[i])
plt.xlim(0, 0.6)
plt.ylim(-0.5, 0.6)
# plt.savefig('svd.pdf')
plt.show()

