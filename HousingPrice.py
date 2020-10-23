import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import SVR

# Input files
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_data.describe()

train_data.head()

test_data.head()

# checking for number of null values
train_data.isnull().sum()

test_data.isnull().sum()

# checking the features of type object and getting the unique values
# for train data
categorical_col = []
for column in train_data.columns:
    if train_data[column].dtype == object and len(train_data[column].unique()) <= 50:
        categorical_col.append(column)

# for test data
categorical_col_test = []
for column in test_data.columns:
    if test_data[column].dtype == object and len(test_data[column].unique()) <= 50:
        categorical_col_test.append(column)

# checking if data type is not a object and getting unique values

numerical_col = []
for column in train_data.columns:
    if train_data[column].dtype != object and len(train_data[column].unique()) <= 50:
        numerical_col.append(column)

numerical_col_test = []
for column in test_data.columns:
    if test_data[column].dtype != object and len(test_data[column].unique()) <= 50:
        numerical_col_test.append(column)

import seaborn as sns
corr = train_data.corr()
plt.figure(figsize = (25,10))
sns.heatmap(corr, annot = True)


# From the corealtion matrix getting values which are least related (less than 30%)

a = abs(corr['SalePrice'])
result = a[a < 0.30]
print(result)

# Making a list of least(less than 30%) co related features. This will be furthur used in drop_features

a = abs(corr['SalePrice'])
result = a[a < 0.30]
print(result)

# Visulazing the distibution of the data for every feature
train_data.hist(edgecolor='blue', linewidth= 2 , figsize=(20, 20));


# Checking missing values in training data set

# Removing columns that have missing values from the training dataset
missing_values_train = [feature for feature in train_data.columns if train_data[feature].isnull().sum() >1]

# Checking missing values in testing dataset

# Removing columns that have missing values from the testing dataset
missing_values_test = [feature for feature in test_data.columns if test_data[feature].isnull().sum() >1]

# Manually dropping features whose missing value % is more than 45%

drop_features = ['Alley', 'PoolQC' , 'Fence' , 'MiscFeature', 'FireplaceQu','ScreenPorch', 'PoolArea','ScreenPorch', 'PoolArea', 'MoSold','3SsnPorch','BsmtFinSF2', 'BsmtHalfBath','MiscVal' ,'LowQualFinSF', 'YrSold', 'OverallCond','MSSubClass' , 'EnclosedPorch' , 'KitchenAbvGr']

train_data.drop(columns = drop_features , inplace = True)

test_data.drop(columns = drop_features , inplace = True)

for feature in drop_features:
    if feature in categorical_col:
        categorical_col.remove(feature)
    if feature in numerical_col:
        numerical_col.remove(feature)

    
for feature in drop_features:
    if feature in categorical_col_test:
        categorical_col_test.remove(feature)
    if feature in numerical_col_test:
        numerical_col_test.remove(feature)

# Extracting the columns which have missing values from the train dataset
missing_values_train = [feature for feature in train_data.columns if train_data[feature].isnull().sum() >1]

missing_values_test = [feature for feature in test_data.columns if test_data[feature].isnull().sum() >1]

for feature in missing_values_train:
    a = np.round(train_data[feature].isnull().mean(),2)
    print(feature, 'has', a *100 , '% of missing values')

for feature in missing_values_test:
    a = np.round(test_data[feature].isnull().mean(),2)
    print(feature, 'has', a *100 , '% of missing values')

# Filling missing values using bfill and ffill

for feature in missing_values_train:
    train_data[feature] = train_data[feature].bfill(axis = 'rows' )
    train_data[feature] = train_data[feature].ffill(axis = 'rows' )

for feature in missing_values_test:
    test_data[feature] = test_data[feature].bfill(axis = 'rows' )
    test_data[feature] = test_data[feature].ffill(axis = 'rows' )


#  **Label encoding**

# Label encoding coverts categorical values into numberical values.

from sklearn import preprocessing
sc=preprocessing.LabelEncoder()

for feature in categorical_col:
    train_data[feature]= sc.fit_transform(train_data[feature].astype(str))

# Converting categorical values into numerical values in test data

# from sklearn import preprocessing
# sc=preprocessing.LabelEncoder()

for feature in categorical_col_test:
    test_data[feature]= sc.fit_transform(test_data[feature].astype(str))

train_data.head()

test_data.head()

# # Splitting the features into independent and dependent variables

X = train_data.drop(['SalePrice'], axis  = 1)
y = train_data['SalePrice']

# Converting float values into int

float_columns = ['BsmtFinSF1', 'BsmtUnfSF','TotalBsmtSF' , 'BsmtFullBath' , 'GarageYrBlt' , 'GarageCars' , 'GarageArea']
for item in float_columns:
    test_data[item].fillna(method='bfill', inplace=True)
    test_data[item].astype(np.int64)

# # Building the model

#Spliting data into test and train

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

# Applying Linear Regression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(test_data)
sc = lr.score(X_test,y_test)
print(sc)

scores = []
cv = KFold(n_splits=10, shuffle=False)

for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    lr.fit(X_train, y_train)
    scores.append(lr.score(X_test, y_test))

print(scores)

lr_pred = lr.predict(test_data)
lr.score(X_test,y_test)

print(lr_pred)

# Calculating r2 score
r2 = r2_score(y_test,lr.predict(X_test))
print('R-Square Score: ',r2*100)

# Submitting files to kaggle

submission_file = pd.DataFrame()
submission_file['Id'] = test_data.Id
submission_file['SalePrice'] = lr_pred

# output file path
submission_file.to_csv('finalsubmission.csv', index=False)