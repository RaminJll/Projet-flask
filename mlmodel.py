import numpy as np
import pandas as pd  
import sklearn as skl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv('FuelConsumption.csv')
le=LabelEncoder()
x = data[['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB',]]
y = data['CO2EMISSIONS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15 , shuffle=False, random_state = 0)

poly_features = PolynomialFeatures(degree=3, include_bias=False)
std_scaler = StandardScaler()
lin_reg = LinearRegression()

# Create
polynomial_regression = make_pipeline(poly_features, std_scaler, lin_reg)
polynomial_regression.fit(x_train, y_train)

print('Correlation Train =', polynomial_regression.score(x_train, y_train))
print('Correlation Test =', polynomial_regression.score(x_test, y_test))

# Predict
y_pred = polynomial_regression.predict(x_test)

import pickle

filename = 'model.pickle'
pickle.dump(polynomial_regression, open(filename, 'wb'))