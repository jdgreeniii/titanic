import pandas as pd
import numpy as np
import matplotlib
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

# Load
data = pd.read_csv('./train.csv')

# First Look
# head = data.head()
# info = data.info()
# des = data.describe()
corr = data.corr()
# hist = data.hist()

# Prepare
target = data['Survived']
num_data = data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
cat_data = data.drop(num_data, axis = 1)

# print(cat_data)


# x = data.info()
# print(x)

# data['Cabin'] = str(data['Cabin'])

# print(x)

cabin_loc_let = []
cabin_loc_num = []


for i in data['Cabin']:
    x = str(i)
    cabin_loc_let.append(x[0])
    if x[1] != 'a':
        y = int(x[1:])
        cabin_loc_num.append(y)
    else:
        cabin_loc_num.append(0)

data['cabin_loc_let'] = cabin_loc_let
data['cabin_loc_num'] = cabin_loc_num
le = preprocessing.LabelEncoder() 
cabin = le.fit_transform(data['cabin_loc_let'])


# print(data['cabin_loc_num'])

# hot/cold encoding
print(len(data))

le = preprocessing.LabelEncoder() 
cabin = le.fit_transform(data['cabin_loc_let'])
cabin = cabin.reshape(-1,1)
enc = OneHotEncoder(sparse = True)
a = enc.fit_transform(cabin).toarray()
# # size of family
# # young and rich
# # replace 0 in cabin_loc_num
# # invert parch

# # corr = data.corr()
# # print(corr)
c = enc.categories_
c = np.asarray(c)
c = c.reshape(-1,1)
print(len(a))
b = le.inverse_transform(c)
print(b)
df_cabin_hot = pd.DataFrame(data = a, columns = b)

new_data = pd.concat([data, df_cabin_hot], axis = 1)
print(new_data)
print(len(new_data))

corr = new_data.corr()
print(corr)