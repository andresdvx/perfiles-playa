import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Leer archivo Excel
archivo = 'perfiles-playa.xlsx'
df = pd.read_excel(archivo)

# Extraer variables
x = df.iloc[:, 0].to_numpy()  # Distancia
fechas = df.columns[1:]       # Nombres de las fechas
y = np.arange(len(fechas))    # Usamos índice como eje Y para fechas

# Construir matriz Z (valores por distancia y fecha)
z = df.iloc[:, 1:].to_numpy().T  # Transpuesta para que cada fila sea una fecha

# Interpolación (aumentar resolución)
res = 100
x_fina = np.linspace(x.min(), x.max(), res)
y_fina = np.linspace(y.min(), y.max(), res)
xg, yg = np.meshgrid(x_fina, y_fina)
zg = np.zeros_like(xg)

# Interpolación bilineal manual
dx = x[1] - x[0]
dy = 1  # ya que Y son índices uniformes

for i in range(res):
    for j in range(res):
        xi, yi = xg[i, j], yg[i, j]
        ix = int((xi - x[0]) / dx)
        iy = int(yi)

        if 0 <= ix < len(x) - 1 and 0 <= iy < len(y) - 1:
            x1, x2 = x[ix], x[ix+1]
            y1, y2 = y[iy], y[iy+1]
            z11 = z[iy, ix]
            z12 = z[iy, ix+1]
            z21 = z[iy+1, ix]
            z22 = z[iy+1, ix+1]

            t = (xi - x1) / (x2 - x1)
            u = (yi - y1) / (y2 - y1)
            zg[i, j] = (1-t)*(1-u)*z11 + t*(1-u)*z12 + (1-t)*u*z21 + t*u*z22
        else:
            zg[i, j] = np.nan

# Gráfica
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xg, yg, zg, cmap=cm.viridis)
ax.set_xlabel('Distancia (m)')
ax.set_ylabel('Índice de Fecha')
ax.set_zlabel('Altura (m)')
ax.set_title('Interpolación de Superficie - Perfiles de Playa')
plt.tight_layout()
plt.show()
