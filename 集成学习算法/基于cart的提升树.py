import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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

# 先选出合适的深度
max_depth = [3, 4, 5, 6]
params1 = {'base_estimator__max_depth': max_depth}
base_model = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                          param_grid=params1, cv=3)
base_model.fit(x_train, y_train)
# best_model = base_model.best_estimator_
base_param = base_model.best_params_
# 返回一个字典类型，取出字典中的值
max_depth = base_param.get('base_estimator__max_depth')
# print(base_param.get(''))

# 机器学习，AdaBoost进行调参数调优
estimator = AdaBoostClassifier()
param_grid = {"n_estimators": [120, 200, 300, 500, 800, 1200]}
# 模型训练次数 5*3  3次交叉验证

estimator = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth)),
                         param_grid=param_grid, cv=3)
#
estimator.fit(x_train, y_train)
n_estimators = estimator.best_params_.get('n_estimators')
AdaBoost2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators)

# 算法在测试集上的拟合
AdaBoost2.fit(x_train, y_train)
# 模型评估
y_pre = AdaBoost2.predict(x_test)
# #
score = AdaBoost2.score(x_test, y_test)
best_estimator = AdaBoost2.base_estimator
print("模型准确率：", score)
print("准确率最高的模型是：", best_estimator)
# 模型准确率为0.7908745247148289
