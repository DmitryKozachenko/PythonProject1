import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Генерація даних для варіанту 10
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)  # Вхідні дані
y = 4 + np.sin(X) + np.random.uniform(-0.6, 0.6, m).reshape(-1, 1)  # Цільові дані

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Поліноміальна регресія (ступінь 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Побудова графіків
plt.figure(figsize=(12, 6))

# Графік лінійної регресії
plt.subplot(1, 2, 1)
plt.scatter(X, y, color="blue", label="Згенеровані дані")
plt.plot(X, y_pred_lin, color="red", linewidth=2, label="Лінійна регресія")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Лінійна регресія")
plt.legend()
plt.grid(True)

# Графік поліноміальної регресії
plt.subplot(1, 2, 2)
plt.scatter(X, y, color="blue", label="Згенеровані дані")
plt.plot(X, y_pred_poly, color="green", linewidth=2, label="Поліноміальна регресія (ступінь 2)")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Поліноміальна регресія (ступінь 2)")
plt.legend()
plt.grid(True)

# Відображення графіків
plt.tight_layout()
plt.show()

# Оцінка якості моделей
print("Лінійна регресія:")
print("MSE:", mean_squared_error(y, y_pred_lin))
print("R2:", r2_score(y, y_pred_lin))

print("\nПоліноміальна регресія (ступінь 2):")
print("MSE:", mean_squared_error(y, y_pred_poly))
print("R2:", r2_score(y, y_pred_poly))
