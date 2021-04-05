import pandas as pd
import numpy as np
import matplotlib
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Load

data = pd.read_csv('./data/train.csv')


# First Look
# head = data.head()
# info = data.info()
# des = data.describe()
# corr = data.corr()
# hist = data.hist()


# split cabin into letter number

cabin_loc_let = []
cabin_loc_num = []

for i in data['Cabin']:
    x = str(i)
    cabin_loc_let.append(x[0])
    if i == 'a':
        cabin_loc_num.append(0)
    else:
        cabin_loc_num.append(x[1:])

data['cabin_loc_let'] = cabin_loc_let
data['cabin_loc_num'] = cabin_loc_num


# label encoder, one hot encoder

le = preprocessing.LabelEncoder() 
cabin = le.fit_transform(data['cabin_loc_let'])
le = preprocessing.LabelEncoder() 
cabin = le.fit_transform(data['cabin_loc_let'])
cabin = cabin.reshape(-1,1)
enc = OneHotEncoder(sparse = True)
a = enc.fit_transform(cabin).toarray()


# add matrix to data frame

c = enc.categories_
c = np.asarray(c)
c = c.reshape(-1,1) 
b = le.inverse_transform(c)
df_cabin_hot = pd.DataFrame(data = a, columns = b)
new_data = pd.concat([data, df_cabin_hot], axis = 1)
corr = new_data.corr()
# print(corr)


# set target, predictors, fill na

new_data = new_data.drop(['Name', 'Sex', 'Embarked', 'Ticket', 'Cabin', 'cabin_loc_let', 'cabin_loc_num'], axis = 1)
mean_age = new_data['Age'].mean()
mean_fare = new_data['Fare'].mean()
new_data['Age'].fillna(value = mean_age, inplace = True)
new_data['Fare'].fillna(value = mean_fare, inplace = True) 
y = new_data['Survived']
X = new_data.drop(['Survived'], axis = 1)


# first model

# clf = LogisticRegressionCV(cv = 10, random_state = 0).fit(X, y)
# clf.predict(X)
# print(clf.score(X, y)) ## -> 0.63585

# X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

# y2 = new_data['Survived'].values
# y2 = np.asarray(y2)
# print(y2)
# regr = RandomForestRegressor(max_depth=2, random_state=0)
# regr.fit(X, y2)
# regr.predict(X)
# print(regr.score(X, y, sample_weight = None)) # -> 0.12269349987378542

#try gradient boosting








# # submit

ss = pd.read_csv('./data/sample_submission.csv')
test = pd.read_csv('./data/test.csv')

cabin_loc_let_test = []
cabin_loc_num_test = []
for i in test['Cabin']:
    x_test = str(i)
    cabin_loc_let_test.append(x_test[0])
    if i == 'a':
        cabin_loc_num_test.append(0)
    else:
        cabin_loc_num_test.append(x_test[1:])

test['cabin_loc_let_test'] = cabin_loc_let_test
test['cabin_loc_num_test'] = cabin_loc_num_test


le_test = preprocessing.LabelEncoder() 
cabin_test = le_test.fit_transform(test['cabin_loc_let_test'])
le_test = preprocessing.LabelEncoder() 
cabin_test = le_test.fit_transform(test['cabin_loc_let_test'])
cabin_test = cabin_test.reshape(-1,1)
enc_test = OneHotEncoder(sparse = True)
a_test = enc_test.fit_transform(cabin_test).toarray()

c_test = enc_test.categories_
c_test = np.asarray(c_test)
c_test = c_test.reshape(-1,1) 
b_test = le_test.inverse_transform(c_test)
df_cabin_hot_test = pd.DataFrame(data = a_test, columns = b_test)
# print(df_cabin_hot_test)
new_data_test = pd.concat([test, df_cabin_hot_test], axis = 1)
# print(new_data_test)
corr = new_data_test.corr()

new_data_test = new_data_test.drop(['Name', 'Sex', 'Embarked', 'Ticket', 'Cabin', 'cabin_loc_let_test', 'cabin_loc_num_test'], axis = 1)
mean_age_test = new_data_test['Age'].mean()
mean_fare_test = new_data_test['Fare'].mean()
new_data_test['Age'].fillna(value = mean_age_test, inplace = True)
new_data_test['Fare'].fillna(value = mean_fare_test, inplace = True) 

X_test = new_data_test

clf_test = LogisticRegressionCV(cv = 10, random_state = 0).fit(X, y)

print('modeled')


y_test = clf_test.predict(X_test)

print('predicted')


y_test = pd.DataFrame(y_test, columns = ['Survived'])
y_test.to_csv('./data/submissions/firstgo.csv', header = True)

sub = pd.read_csv('./data/submissions/firstgo.csv')
sub['PassengerId'] = sub['Unnamed: 0'] + 100000
print(sub['PassengerId'])
sub = sub.drop(['Unnamed: 0'], axis = 1)
sub = sub.reindex(columns = ['PassengerId', 'Survived'])
sub = sub.set_index(['PassengerId'])

sub.to_csv('./data/submissions/firstgo.csv', header = True)
print('done')
# print(sub['Unnamed: 0'])





# # size of family
# # young and rich
# # replace 0 in cabin_loc_num
# # invert parch