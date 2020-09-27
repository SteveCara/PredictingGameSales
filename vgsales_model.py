import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('vgsalesclean.csv', index_col=0)
# print(df.head())

# choose relevant columns

# rank wont influence EU sales of racing games
#  Platform, Year, Genre, Publsiher are the key independent variables
#  EU_Sales is the target variable

features = ['Platform', 'Year', 'Genre', 'Publisher', 'EU_Sales']
df_model = df[features]

# get dummy data
df_dum = pd.get_dummies(df_model)

# set X and y
X = df_dum.drop('EU_Sales', axis=1)
y = df_dum['EU_Sales'].values

# train, test, split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# # multiple linear regression
lm = LinearRegression()
print(np.mean(cross_val_score(lm, X_train, y_train,
                              scoring='neg_mean_absolute_error', cv=10)))

# this does not perform well as most of the data is in a 'sparse matrix' as it is categorical and therefore much of the information comes from 1 or 0 in dummy data
# not many nice highly correlated patterns

# lasso regression
lm_l = Lasso()
print(np.mean(cross_val_score(lm_l, X_train, y_train,
                              scoring='neg_mean_absolute_error', cv=3)))

# random forest
rf = RandomForestRegressor()
print(np.mean(cross_val_score(rf, X_train, y_train,
                              scoring='neg_mean_absolute_error', cv=3)))

# hyperparameter tuning
# use GridSearchCV to improve the model
