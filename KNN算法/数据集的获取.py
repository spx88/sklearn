from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, fetch_20newsgroups
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# 小数据集的获取
iris = load_iris()
# 鸢尾花数据集
# 类别 3
# 特征 4
# 样本数量 150
# 每个类别数量 50
# print(iris)
# 大数据集的获取
# news = fetch_20newsgroups()
# print(news)

# # 数据集属性描述
# print("数据集特征值名字是\n", iris.feature_names)
# print("数据集特征值是\n", iris.data)
# print("数据集的目标值是\n", iris["target"])
# print("数据集目标值名字是\n", iris.target_names)
# print("数据集的描述\n", iris.DESCR)
#
# 3.数据集的可视化
# 这里是数据集的特征和特征名字传入
iris_d = pd.DataFrame(data=iris.data, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
print(iris_d)
iris_d["target"] = iris.target


def iris_plot(data, colo1, colo2):
    sns.lmplot(x=colo1, y=colo2, data=data, hue="target", fit_reg=False)
    plt.title("鸢尾花数据显示")
    plt.show()


# 数据集的划分
# 测试集0.2
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
print("训练集的特征值", x_train)
print("训练接的目标值", y_train)
print("测试集的特征值", x_test)
print("测试集的目标值", y_test)


# iris_plot(iris_d, 'sepal width', 'petal length', )
