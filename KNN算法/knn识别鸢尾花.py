from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. 获取数据集
iris = load_iris()

# 2.数据基本处理, random_state随机数种子
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

# 3.特征工程：标准化

transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4.模型训练
# 4.1 实例化一个估计器
estimator = KNeighborsClassifier()
# 4.2模型调优 -- 交叉验证，网格搜索
# 超参数K值得设定
param_grid = {"n_neighbors": [1, 3, 5, 7]}
# cv就是选择交叉验证的折数
estimator = GridSearchCV(estimator, param_grid=param_grid, cv=5)

# 4.3模型训练
estimator.fit(x_train, y_train)
# 5.模型评估
# 对比真实值和预测值
y_predict = estimator.predict(x_test)
print("预测结果为\n", y_predict)
print("真实值为\n", y_test)
print("比对真实值和预测值：\n", y_predict == y_test)
# 或者直接计算准确率
score = estimator.score(x_test, y_test)
print("准确率为：\n", score)

# 5.3 查看交叉验证，网格搜索的一些属性
print("在交叉验证中得到的最好结果", estimator.best_score_)  # 也就是五次交叉验证中最好的
print("交叉验证得到的最好模型", estimator.best_estimator_)  # 四个超参数五折交叉中验证的最好模型
