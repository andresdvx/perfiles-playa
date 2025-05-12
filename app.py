import pandas as pd
import numpy as np
from scipy.interpolate import lagrange, CubicSpline
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import xlsxwriter

# Leer archivo
df = pd.read_excel("analisis.xlsx", sheet_name="Hoja1")
fechas = df.columns[1:]
x = df["Distancia"].to_numpy()
x_interp = np.linspace(x.min(), x.max(), 200)

def newton_divided_diff(x, y):
    n = len(y)
    coef = np.copy(y)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1]) / (x[j:n] - x[j - 1])
    return coef

def newton_polynomial(x_data, coef, x_interp):
    result = np.zeros_like(x_interp)
    for i in range(len(coef)):
        term = coef[i]
        for j in range(i):
            term *= (x_interp - x_data[j])
        result += term
    return result

def calc_metrics(y_ref_interp, y_pred):
    rmse = np.sqrt(mean_squared_error(y_ref_interp, y_pred))
    corr, _ = pearsonr(y_ref_interp, y_pred)
    return rmse, corr

# Crear archivo Excel
workbook = xlsxwriter.Workbook("perfiles_comparativos_completo.xlsx")
resumen_ws = workbook.add_worksheet("Resumen Comparativo")
curvas_ws = workbook.add_worksheet("Curvas Interpoladas")

# Encabezados
resumen_ws.write_row(0, 0, ["Fecha", "Método", "RMSE", "Correlación", "Mejor"])
curvas_ws.write(0, 0, "Distancia")
for i, x_val in enumerate(x_interp, start=1):
    curvas_ws.write(i, 0, x_val)

row_resumen = 1
col_curvas = 1

for fecha in fechas:
    y = df[fecha].to_numpy()
    y_ref_interp = np.interp(x_interp, x, y)

    lagrange_poly = lagrange(x, y)
    y_lagrange = lagrange_poly(x_interp)

    newton_coef = newton_divided_diff(x, y)
    y_newton = newton_polynomial(x, newton_coef, x_interp)
    

    spline = CubicSpline(x, y)
    y_spline = spline(x_interp)

    poly_coeffs = np.polyfit(x, y, 5)
    y_regresion = np.polyval(poly_coeffs, x_interp)

    resultados = {
        "Lagrange": (y_lagrange, *calc_metrics(y_ref_interp, y_lagrange)),
        "Newton": (y_newton, *calc_metrics(y_ref_interp, y_newton)),
        "Spline": (y_spline, *calc_metrics(y_ref_interp, y_spline)),
        "Regresión": (y_regresion, *calc_metrics(y_ref_interp, y_regresion))
    }

    mejor = min(resultados.items(), key=lambda item: item[1][1])[0]

    for metodo, (y_calc, rmse, corr) in resultados.items():
        resumen_ws.write(row_resumen, 0, str(fecha.date()))
        resumen_ws.write(row_resumen, 1, metodo)
        resumen_ws.write(row_resumen, 2, rmse)
        resumen_ws.write(row_resumen, 3, corr)
        resumen_ws.write(row_resumen, 4, "Sí" if metodo == mejor else "")
        row_resumen += 1

        curvas_ws.write(0, col_curvas, f"{fecha.date()} - {metodo}")
        for i, val in enumerate(y_calc, start=1):
            curvas_ws.write(i, col_curvas, val)
        col_curvas += 1

workbook.close()
