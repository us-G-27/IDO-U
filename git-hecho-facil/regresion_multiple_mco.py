import numpy as np

# ==========================================
# 1. DATOS DE MUESTRA
# ==========================================
# Variables explicativas (X1, X2):
# X1 = Gasto en publicidad
# X2 = Número de agentes / vendedores
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

# Variable de respuesta (Y): Ventas obtenidas
Y = np.array([45, 60, 52, 78, 92, 70, 110, 85], dtype=float)

n = len(Y)

# ==========================================
# 2. CONSTRUCCIÓN DE LA MATRIZ DE DISEÑO (X)
# ==========================================
# Se agrega el vector de unos para estimar el intercepto (beta_0)
unos = np.ones((n, 1))
X = np.hstack((unos, X_raw))  # Dimensión (n x k+1)
k = X.shape[1] - 1            # Número de variables regresoras (sin intercepto)

# ==========================================
# 3. ESTIMACIÓN MCO: Betas = (X^T * X)^(-1) * X^T * Y
# ==========================================
XtX = np.dot(X.T, X)
XtY = np.dot(X.T, Y)
betas = np.linalg.inv(XtX).dot(XtY)

# ==========================================
# 4. VALORES AJUSTADOS, RESIDUOS Y MÉTRICAS
# ==========================================
Y_hat = np.dot(X, betas)
residuos = Y - Y_hat

# Sumas de cuadrados
SRC = np.sum(residuos**2)                  # Suma de Residuos al Cuadrado
STC = np.sum((Y - np.mean(Y))**2)          # Suma Total de Cuadrados
SEC = STC - SRC                            # Suma Explicada de Cuadrados

# Coeficientes de determinación
R2 = 1 - (SRC / STC)
R2_adj = 1 - ((SRC / (n - k - 1)) / (STC / (n - 1)))

# Varianza del error y matriz de varianzas-covarianzas de los estimadores
sigma2_hat = SRC / (n - k - 1)
var_cov_betas = sigma2_hat * np.linalg.inv(XtX)
errores_estandar = np.sqrt(np.diagonal(var_cov_betas))

# ==========================================
# 5. SALIDA DE RESULTADOS
# ==========================================
print("=========================================================")
print("        REGRESIÓN LINEAL MÚLTIPLE (MCO MATRICIAL)        ")
print("=========================================================")
print(f"Beta_0 (Intercepto) : {betas[0]:10.4f} | EE: {errores_estandar[0]:.4f}")
print(f"Beta_1 (Publicidad) : {betas[1]:10.4f} | EE: {errores_estandar[1]:.4f}")
print(f"Beta_2 (Vendedores) : {betas[2]:10.4f} | EE: {errores_estandar[2]:.4f}")
print("---------------------------------------------------------")
print(f"R² (Coef. de Determinación): {R2:.4f}")
print(f"R² Ajustado                : {R2_adj:.4f}")
print(f"S² (Varianza del Error)    : {sigma2_hat:.4f}")
print(f"SRC (Suma Residuos Cuad.)  : {SRC:.4f}")
print("=========================================================")