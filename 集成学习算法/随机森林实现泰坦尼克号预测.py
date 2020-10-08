import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

titan = pd.read_csv("./titanic.txt")

# 这里我们只取pclass age sex 为特征值，取survived为目标值
x = titan[["pclass", "age", "sex"]]
y = titan["survived"]

# 缺失值处理
x['age'].fillna(value=titan["age"].mean(), inplace=True)
# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.2)
x_train = x_train.to_dict(orient="records")
x_test = x_test.to_dict(orient="records")
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)
# 机器学习，随机森林，bagging+cart决策树
estimator = RandomForestClassifier()
# n_estimators森林里树木的数量,max_depth 树的最大深度
param_grid = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
# 模型训练次数 5*5*3  五个超参数 五个超参数 3次交叉验证
estimator = GridSearchCV(estimator, param_grid=param_grid, cv=3)

estimator.fit(x_train, y_train)
# 模型评估
y_pre = estimator.predict(x_test)

score = estimator.score(x_test, y_test)
best_estimator = estimator.best_estimator_
best_param = estimator.best_params_
print("模型准确率：", score)
print("准确率最高的模型是：", best_estimator)
print("最优超参数：", best_param)
