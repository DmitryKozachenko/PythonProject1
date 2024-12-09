import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Генерація даних (варіант 10)
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 4 + np.sin(X) + np.random.uniform(-0.6, 0.6, m).reshape(-1, 1)

# Функція для побудови кривих навчання
def compute_learning_curves(model, X, y, degree=None):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_errors, test_errors = [], []

    for train_size in train_sizes:
        m_train = int(len(X) * train_size)
        X_train, y_train = X[:m_train], y[:m_train]

        if degree:
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_train = poly_features.fit_transform(X_train)
            X_poly = poly_features.transform(X)
        else:
            X_poly = X

        model.fit(X_train, y_train)
        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_poly)

        train_errors.append(mean_squared_error(y_train, y_train_predict))
        test_errors.append(mean_squared_error(y, y_test_predict))

    return train_sizes, train_errors, test_errors

# Створення моделей
lin_reg = LinearRegression()
poly_reg = LinearRegression()

# Обчислення кривих навчання
lin_train_sizes, lin_train_errors, lin_test_errors = compute_learning_curves(lin_reg, X, y)
poly_train_sizes, poly_train_errors, poly_test_errors = compute_learning_curves(poly_reg, X, y, degree=2)

# Побудова графіків
plt.figure(figsize=(10, 6))

# Лінійна регресія
plt.plot(lin_train_sizes, lin_train_errors, "r-+", linewidth=2, label="Лінійна: Помилка на тренуванні")
plt.plot(lin_train_sizes, lin_test_errors, "r-", linewidth=2, label="Лінійна: Помилка на тестуванні")

# Поліноміальна регресія (ступінь 2)
plt.plot(poly_train_sizes, poly_train_errors, "b-+", linewidth=2, label="Поліном 2-го ст.: Помилка на тренуванні")
plt.plot(poly_train_sizes, poly_test_errors, "b-", linewidth=2, label="Поліном 2-го ст.: Помилка на тестуванні")

# Оформлення графіка
plt.xlabel("Розмір тренувальної вибірки")
plt.ylabel("MSE")
plt.title("Криві навчання для лінійної та поліноміальної регресії")
plt.legend()
plt.grid(True)
plt.show()
