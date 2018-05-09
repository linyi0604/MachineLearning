import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

selected_features = ["Pclass", "Sex", "Age", "SibSp", "Embarked", "Parch", "Fare"]
x_train = train[selected_features]
y_train = train["Survived"]
x_test = test[selected_features]

x_train["Age"].fillna(x_train["Age"].mean(), inplace=True)
x_train["Embarked"].fillna("S", inplace=True)
x_test["Age"].fillna(x_test["Age"].mean(), inplace=True)
x_test["Fare"].fillna(x_test["Fare"].mean(), inplace=True)

dict_vec = DictVectorizer(sparse=False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient="record"))
x_test = dict_vec.transform(x_test.to_dict(orient="record"))

# print(dict_vec.feature_names_)
# ['Age', 'Embarked=C', 'Embarked=Q', 'Embarked=S', 'Fare', 'Parch', 'Pclass', 'Sex=female', 'Sex=male', 'SibSp']
rfc = RandomForestClassifier()
xgbc = XGBClassifier()

rfc_score = cross_val_score(rfc, x_train, y_train, cv=5).mean() # 0.8048504727484737
xgbc_score = cross_val_score(xgbc, x_train, y_train, cv=5).mean()   # 0.81824559798311

# 使用随机森林进行预测
rfc.fit(x_train, y_train)
rfc_y_predict = rfc.predict(x_test)
rfc_submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": rfc_y_predict
})
# 预测结果保存
rfc_submission.to_csv("./predict/rfc_submission.csv", index=False)


# 使用xgbc进行预测
xgbc.fit(x_train, y_train)
xgbc_y_predict = xgbc.predict(x_test)
# 保存预测结果
xgbc_submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": xgbc_y_predict
})
xgbc_submission.to_csv("./predict/xgbc_submission.csv", index=False)

# 使用并行搜索 寻找更好的超参数组合
params = {
    "max_depth": [i for i in range(2, 7)],
    "n_estimators": [i for i in range(100, 1100, 200)],
    "learning_rate": [0.05, 0.1, 0.25, 0.5, 1.0]
}


xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
xgbc_best_y_predict = gs.predict(x_test)
xgbc_best_submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": xgbc_best_y_predict
})
xgbc_best_submission.to_csv("./predict/xgbc_best_submission.csv", index=False)


xgbc_best2 = XGBClassifier()
selected_features2 = ["Pclass", "Sex", "Age", "Embarked"]
x_train = train[selected_features2]
x_test = test[selected_features2]
x_train = dict_vec.fit_transform(x_train.to_dict(orient="record"))
x_test = dict_vec.transform(x_test.to_dict(orient="record"))

xgbc_score = cross_val_score(xgbc, x_train, y_train, cv=5).mean()   # 0.820454913793134

xgbc.fit(x_train, y_train)
xgbc2_y_predict = xgbc.predict(x_test)
xgbc2_submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": xgbc2_y_predict
})
xgbc2_submission.to_csv("./predict/xgbc2_submission.csv", index=False)


gs = GridSearchCV(xgbc_best2, params, n_jobs=-1, cv=5, verbose=1)
gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
'''
0.8237934904601572
{'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05}
'''
xgbc_best2_y_predict = gs.predict(x_test)
xgbc_best2_submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": xgbc_best2_y_predict
})
xgbc_best2_submission.to_csv("./predict/xgbc_best2_submission.csv", index=False)

