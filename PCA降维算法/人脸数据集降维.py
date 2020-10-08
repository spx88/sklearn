from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

# 导入数据，并且探索一下子
faces = fetch_lfw_people(min_faces_per_person=60)
faces.images.shape  # (1348, 64, 47)  1348张图片，每张64*47
faces.data.shape  # (1348, 2914)  这是把上面的后两维进行了合并，共2914个特征（像素点）
# 下面我们先可视化一下子这些图片，看看长什么样
fig, axes = plt.subplots(3, 8, figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})

for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i, :, :], cmap='gray')
