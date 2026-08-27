import numpy as np

# 1. Matriz de variables explicativas (X1, X2) y vector dependiente (Y)
# Ejemplo: X1 = Gasto en Publicidad, X2 = Número de Vendedores, Y = Ventas
X_raw = np.array([
    [10, 2],
    [15, 3],
    [12, 2],
    [20, 5],
    [25, 6],
    [18, 4],
    [30, 7],
    [22, 5]
], dtype=float)

Y = np.array([45, 60, 52, 78, 92, 70, 110, 85], dtype=float)

n = len(Y)

# 2. Agregar la columna de 1s para el intercepto (beta_0)
unos = np.ones((n, 1))
X = np.hstack((unos, X_raw))  # Dimensión: (n x (k+1))
k = X.shape[1] - 1  # Número de regresores excluyendo el intercepto

# 3. Estimación de Betas: (X^T X)^(-1) X^T Y
XtX = np.dot(X.T, X)
XtY = np.dot(X.T, Y)
betas = np.linalg.inv(XtX).dot(XtY)

# 4. Valores ajustados y análisis de varianza
Y_hat = np.dot(X, betas)
residuos = Y - Y_hat
SRC = np.sum(residuos**2)  # Suma de residuos al cuadrado
STC = np.sum((Y - np.mean(Y))**2)  # Suma total de cuadrados
R2 = 1 - (SRC / STC)
R2_adj = 1 - ((SRC / (n - k - 1)) / (STC / (n - 1)))

# Varianza del error y matriz de covarianzas de los estimadores
sigma2_hat = SRC / (n - k - 1)
var_cov_betas = sigma2_hat * np.linalg.inv(XtX)
errores_estandar = np.sqrt(np.diagonal(var_cov_betas))

print("=== REGRESIÓN LINEAL MÚLTIPLE (MCO MATRICIAL) ===")
print(f"Beta_0 (Intercepto): {betas[0]:.4f} | EE: {errores_estandar[0]:.4f}")
print(f"Beta_1:              {betas[1]:.4f} | EE: {errores_estandar[1]:.4f}")
print(f"Beta_2:              {betas[2]:.4f} | EE: {errores_estandar[2]:.4f}")
print("-------------------------------------------------")
print(f"R^2:                 {R2:.4f}")
print(f"R^2 Ajustado:        {R2_adj:.4f}")
print(f"Sigma^2 estimado:    {sigma2_hat:.4f}")