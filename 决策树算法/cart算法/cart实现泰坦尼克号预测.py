import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn.metrics

titan = pd.read_csv("../cart算法/titanic.txt")

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
# 机器学习，决策树，c4.5算法,传入参数不同算法不同，默认是gini也就是cart算法
estimator = DecisionTreeClassifier()
# 超参数设置
param_grid = {"max_depth": [5, 7, 8]}
estimator = GridSearchCV(estimator, param_grid=param_grid, cv=3)
estimator.fit(x_train, y_train)
# 模型评估
y_pre = estimator.predict(x_test)

score = estimator.score(x_test, y_test)
best_param = estimator.best_params_
print("最好的参数：", best_param)
print("预测准确率：", score)
