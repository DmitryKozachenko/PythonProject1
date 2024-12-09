import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантаження даних про діабет
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розбиття даних на навчальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Створення та навчання лінійного регресора
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Прогнозування на тестових даних
y_pred = regr.predict(X_test)

# Розрахунок коефіцієнтів регресії та метрик
print("Coefficients:", regr.coef_)
print("Intercept:", regr.intercept_)
print("Mean squared error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean absolute error (MAE):", mean_absolute_error(y_test, y_pred))
print("Coefficient of determination (R^2):", r2_score(y_test, y_pred))

# Побудова графіка
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0), label='Прогноз проти істинних значень')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ідеальна лінія')
plt.xlabel('Виміряно')
plt.ylabel('Передбачено')
plt.title('Лінійна регресія на даних про діабет')
plt.legend()
plt.show()
