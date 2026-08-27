import numpy as np

# 1. Datos de muestra (por ejemplo: Horas de estudio vs Calificación)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([2.5, 3.8, 5.1, 5.9, 7.2, 8.0, 8.9, 10.1, 11.3, 12.5], dtype=float)

n = len(x)

# 2. Estimadores por MCO (Formulación escalar)
# beta_1 = Cov(x, y) / Var(x)
# beta_0 = y_media - beta_1 * x_media
x_bar = np.mean(x)
y_bar = np.mean(y)

beta_1 = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar)**2)
beta_0 = y_bar - beta_1 * x_bar

# 3. Valores ajustados y residuos
y_hat = beta_0 + beta_1 * x
residuos = y - y_hat

# 4. Bondad de ajuste (R^2)
ss_tot = np.sum((y - y_bar)**2)
ss_res = np.sum(residuos**2)
r_cuadrado = 1 - (ss_res / ss_tot)

print("=== REGRESIÓN LINEAL SIMPLE (MCO) ===")
print(f"Ecuación estimada: y_hat = {beta_0:.4f} + {beta_1:.4f} * x")
print(f"Coeficiente R^2: {r_cuadrado:.4f}")
print(f"Suma de Residuos al Cuadrado (SRC): {ss_res:.4f}")