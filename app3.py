import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# FUNCIONES MANUALES DE INTERPOLACIÓN
# ------------------------------

def lagrange_interpolation(x, y, x_eval):
    n = len(x)
    result = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x_eval - x[j]) / (x[i] - x[j])
        result += term
    return result

def newton_divided_differences(x, y):
    n = len(x)
    coef = np.copy(y)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef

def newton_interpolation(x, coef, x_eval):
    n = len(coef)
    result = coef[-1]
    for k in range(2, n+1):
        result = coef[-k] + (x_eval - x[-k]) * result
    return result

def spline_coefficients(x, y):
    n = len(x)
    a = y.copy()
    h = np.diff(x)
    alpha = [0] * (n-1)
    for i in range(1, n-1):
        alpha[i] = (3/h[i])*(a[i+1]-a[i]) - (3/h[i-1])*(a[i]-a[i-1])
    
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    
    for i in range(1, n-1):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]
    
    b = np.zeros(n-1)
    c = np.zeros(n)
    d = np.zeros(n-1)
    
    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (a[j+1]-a[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j]) / (3*h[j])
        
    return a, b, c, d

def spline_evaluation(x, coefs, x_eval):
    a, b, c, d = coefs
    n = len(a)
    for i in range(n-1):
        if x[i] <= x_eval <= x[i+1]:
            dx = x_eval - x[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return None

def regresion_polinomial(x, y, grado, x_eval):
    X = np.vander(x, grado+1, increasing=True)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    powers = np.array([x_eval**i for i in range(grado+1)])
    return np.dot(coef, powers)

def calcular_error(y_real, y_pred):
    return np.mean(np.abs(y_real - y_pred))

def calcular_correlacion(y_real, y_pred):
    return np.corrcoef(y_real, y_pred)[0,1]

# ------------------------------
# PROGRAMA PRINCIPAL
# ------------------------------

# 1. Leer Excel
archivo = 'perfiles-playa.xlsx'  # Tu archivo
data = pd.read_excel(archivo)

# Variables
x = data.iloc[:, 0].to_numpy()  # primera columna: distancia
columnas_fechas = data.columns[1:]  # resto de columnas: fechas

# 2. Procesar cada fecha
for fecha in columnas_fechas:
    print(f"Procesando fecha {fecha}...")
    
    y = data[fecha].to_numpy()

    # Preparar puntos de evaluación
    x_fino = np.linspace(x.min(), x.max(), 200)

    # Lagrange
    y_lagrange = np.array([lagrange_interpolation(x, y, xi) for xi in x_fino])

    # Newton
    coef_newton = newton_divided_differences(x, y)
    y_newton = np.array([newton_interpolation(x, coef_newton, xi) for xi in x_fino])

    # Spline
    coef_spline = spline_coefficients(x, y)
    y_spline = np.array([spline_evaluation(x, coef_spline, xi) for xi in x_fino])

    # Regresión polinomial
    grado = min(len(x)-1, 5)
    y_regresion = np.array([regresion_polinomial(x, y, grado, xi) for xi in x_fino])

    # Valores reales en x_fino para comparación (interpolamos linealmente real vs x_fino)
    y_real = np.interp(x_fino, x, y)

    # 3. Evaluar errores
    errores = {}
    correlaciones = {}

    errores['Lagrange'] = calcular_error(y_real, y_lagrange)
    correlaciones['Lagrange'] = calcular_correlacion(y_real, y_lagrange)

    errores['Newton'] = calcular_error(y_real, y_newton)
    correlaciones['Newton'] = calcular_correlacion(y_real, y_newton)

    errores['Spline'] = calcular_error(y_real, y_spline)
    correlaciones['Spline'] = calcular_correlacion(y_real, y_spline)

    errores['Regresion'] = calcular_error(y_real, y_regresion)
    correlaciones['Regresion'] = calcular_correlacion(y_real, y_regresion)

    # 4. Graficar
    plt.figure(figsize=(12,6))
    plt.plot(x, y, 'ko-', label='Datos reales', markersize=5)
    plt.plot(x_fino, y_lagrange, label='Lagrange')
    plt.plot(x_fino, y_newton, label='Newton')
    plt.plot(x_fino, y_spline, label='Spline Cúbico')
    plt.plot(x_fino, y_regresion, label='Regresión Polinomial')
    plt.title(f'Interpolación para la fecha {fecha}')
    plt.xlabel('Distancia (m)')
    plt.ylabel('Altura (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5. Mostrar resultados
    print(f"Errores para {fecha}:")
    for metodo in errores:
        print(f"{metodo}: Error medio = {errores[metodo]:.5f}, Correlacion = {correlaciones[metodo]:.5f}")


    mejor_metodo = min(errores, key=errores.get)
    print(f"\n✨ Mejor método para {fecha}: {mejor_metodo}\n")
