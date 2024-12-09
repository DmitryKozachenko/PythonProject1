import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Завантаження даних
input_file = 'data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка даних на навчальний і тестовий набори
num_training = int(0.8 * len(X))
num_test = len(X) - num_training
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Створення та навчання лінійного регресора
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

# Прогнозування на тестових даних
y_test_pred = linear_regressor.predict(X_test)

# Оцінка якості лінійної моделі
print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Створення поліноміальної регресії ступеня 10
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
X_test_transformed = polynomial.transform(X_test)

# Навчання поліноміальної моделі
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

# Прогнозування за поліноміальною моделлю
y_test_pred_poly = poly_linear_model.predict(X_test_transformed)

# Порівняння лінійної та поліноміальної регресій
print("\nComparison:")
print("Linear regression MAE =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Polynomial regression MAE =", round(sm.mean_absolute_error(y_test, y_test_pred_poly), 2))

# Приклад прогнозу для конкретної точки
datapoint = [[7.75, 6.35, 5.56]]
datapoint_transformed = polynomial.transform(datapoint)
print("\nPrediction for datapoint using linear regression:", linear_regressor.predict(datapoint))
print("Prediction for datapoint using polynomial regression:", poly_linear_model.predict(datapoint_transformed))