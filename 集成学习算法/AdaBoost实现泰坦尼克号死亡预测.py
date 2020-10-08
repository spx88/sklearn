import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import AdaBoostClassifier

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
# 机器学习，AdaBoost
estimator = AdaBoostClassifier()

param_grid = {"n_estimators": [120, 200, 300, 500, 800, 1200]}
# 模型训练次数 5*3  五个超参数 3次交叉验证
estimator = GridSearchCV(estimator, param_grid=param_grid, cv=3)

estimator.fit(x_train, y_train)
# 模型评估
y_pre = estimator.predict(x_test)

score = estimator.score(x_test, y_test)
best_estimator = estimator.best_estimator_
print("模型准确率：", score)
print("准确率最高的模型是：", best_estimator)
# 模型准确率为0.7908745247148289
