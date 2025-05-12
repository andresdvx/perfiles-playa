import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline, BarycentricInterpolator
from numpy.polynomial import Polynomial

# 1. Leer los datos desde un archivo Excel
data = pd.read_excel('analisis.xlsx')  # Cambia al nombre de tu archivo

# Supongamos que tus datos son columnas 'x' y 'y'
x = data['x'].to_numpy()
y = data['y'].to_numpy()

# Crear puntos finos para graficar interpolaciones
x_fine = np.linspace(x.min(), x.max(), 500)

# 2. Aplicar métodos de interpolación

# Lagrange
poly_lagrange = lagrange(x, y)
y_lagrange = poly_lagrange(x_fine)

# Trazadores cúbicos (Splines)
spline = CubicSpline(x, y)
y_spline = spline(x_fine)

# Newton (utilizamos BarycentricInterpolator para aproximar el método)
newton_interp = BarycentricInterpolator(x, y)
y_newton = newton_interp(x_fine)

# Regresión polinomial (grado mejor ajustado)
grado = min(len(x) - 1, 5)  # Grado máximo 5 o menos si hay pocos puntos
coefs = np.polyfit(x, y, grado)
poly_regresion = np.poly1d(coefs)
y_regresion = poly_regresion(x_fine)

# 3. Evaluar error y correlación

def calcular_metricas(y_real, y_predicho):
    error = np.mean(np.abs(y_real - y_predicho))  # Error medio absoluto
    correlacion = np.corrcoef(y_real, y_predicho)[0, 1]  # Coeficiente de correlación
    return error, correlacion

# Calculamos
error_lagrange, corr_lagrange = calcular_metricas(y, poly_lagrange(x))
error_spline, corr_spline = calcular_metricas(y, spline(x))
error_newton, corr_newton = calcular_metricas(y, newton_interp(x))
error_regresion, corr_regresion = calcular_metricas(y, poly_regresion(x))

# 4. Mostrar resultados
print("\nResultados de comparación:")
print(f"Lagrange:      Error medio = {error_lagrange:.5f}, Correlacion = {corr_lagrange:.5f}")
print(f"Spline Cúbico: Error medio = {error_spline:.5f}, Correlacion = {corr_spline:.5f}")
print(f"Newton:        Error medio = {error_newton:.5f}, Correlacion = {corr_newton:.5f}")
print(f"Regresión Polinomial: Error medio = {error_regresion:.5f}, Correlacion = {corr_regresion:.5f}")

# 5. Determinar el mejor método
metodos = ['Lagrange', 'Spline', 'Newton', 'Regresión']
errores = [error_lagrange, error_spline, error_newton, error_regresion]
correlaciones = [corr_lagrange, corr_spline, corr_newton, corr_regresion]

# Criterio: menor error y mayor correlación
mejor_error = min(errores)
mejor_correlacion = max(correlaciones)

mejores_metodos = [
    nombre for nombre, err, corr in zip(metodos, errores, correlaciones)
    if np.isclose(err, mejor_error) and np.isclose(corr, mejor_correlacion)
]

if len(mejores_metodos) == 1:
    print(f"\n✨ El mejor método fue: {mejores_metodos[0]}")
else:
    print(f"\n🔍 Hubo un empate entre los mejores métodos: {', '.join(mejores_metodos)}")


# 6. Graficar todos los perfiles
plt.figure(figsize=(14, 6))
plt.plot(x, y, 'o', label='Datos originales', markersize=8)
plt.plot(x_fine, y_lagrange, label='Lagrange')
plt.plot(x_fine, y_spline, label='Spline Cúbico')
plt.plot(x_fine, y_newton, label='Newton')
plt.plot(x_fine, y_regresion, label='Regresión Polinomial')

plt.title('Comparación de Métodos de Interpolación')
plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.legend()
plt.grid(True)

plt.show()