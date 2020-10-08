import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data

# X的归一化
X_norm = StandardScaler().fit_transform(X)
X_norm.mean(axis=0)

# 然后使用,n_components决定降到几维，
pca = PCA(n_components=2)
X_new = pca.fit_transform(X_norm)

"""查看PCA的一些属性"""
print(pca.explained_variance_)  # 属性可以查看降维后的每个特征向量上所带的信息量大小（可解释性方差的大小）
print(pca.explained_variance_ratio_)  # 查看降维后的每个新特征的信息量占原始数据总信息量的百分比
print(pca.explained_variance_ratio_.sum())  # 降维后信息保留量
