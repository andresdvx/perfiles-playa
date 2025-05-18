import numpy as np
import matplotlib.pyplot as plt
import math

# Datos internos (puedes modificar estos valores según tu caso)
x = [
    0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5,
    5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5,
    10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5,
    15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5,
    20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5,
    25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5,
    30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5,
    35, 35.5, 36, 36.5, 37, 37.5, 38, 38.5, 39, 39.5,
    40, 40.5, 41, 41.5, 42, 42.5, 43, 43.5, 44, 44.5,
    45, 45.5
];

y = [
    0.7000, 0.6088, 0.5337, 0.4732, 0.4258, 0.3900, 0.3643, 0.3471, 0.3370, 0.3324,
    0.3318, 0.3336, 0.3365, 0.3387, 0.3389, 0.3355, 0.3270, 0.2656, 0.2377, 0.2129,
    0.1910, 0.1720, 0.1557, 0.1420, 0.1308, 0.1220, 0.1154, 0.1110, 0.0502, 0.1157,
    0.1543, 0.1707, 0.1691, 0.1534, 0.1270, 0.0931, 0.0544, 0.0134, -0.0280, -0.0681,
    -0.1057, -0.1398, -0.1700, -0.1960, -0.2180, -0.2367, -0.2529, -0.2680, -0.2837, -0.3020,
    -0.3254, -0.2543, -0.2565, -0.2657, -0.2801, -0.2980, -0.3175, -0.3369, -0.3543, -0.3680,
    -0.3762, -0.3771, -0.3689, -0.3499, -0.3182, -0.2720, -0.2740, -0.2745, -0.2744, -0.2739,
    -0.2731, -0.2721, -0.2710, -0.2699, -0.2689, -0.2681, -0.2676, -0.2675, -0.2680, -0.2691,
    -0.2710, -0.2737, -0.2774, -0.2821, -0.2880, -0.3430, -0.3337, -0.3287, -0.3277, -0.3302,
    -0.3358, -0.3443
]


def graficar(x, y, func, titulo):
    x_line = np.linspace(min(x), max(x), 200)
    y_line = [func(xi) for xi in x_line]

    plt.plot(x, y, 'o', label="Datos reales")
    plt.plot(x_line, y_line, label=titulo)
    plt.title(titulo)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def regresion_lineal(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(xi ** 2 for xi in x)
    sum_xy = sum(x[i] * y[i] for i in range(n))

    pendiente = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    intercepto = (sum_y - pendiente * sum_x) / n

    def f(xi):
        return pendiente * xi + intercepto

    return f

def regresion_polinomial(x, y, grado):
    n = len(x)
    A = [[sum(xi ** (i + j) for xi in x) for j in range(grado + 1)] for i in range(grado + 1)]
    B = [sum(y[k] * x[k] ** i for k in range(n)) for i in range(grado + 1)]
    coef = resolver_sistema(A, B)

    def f(xi):
        return sum(coef[j] * xi ** j for j in range(grado + 1))

    return f

def resolver_sistema(A, B):
    n = len(B)
    for i in range(n):
        divisor = A[i][i]
        for j in range(i, n):
            A[i][j] /= divisor
        B[i] /= divisor
        for k in range(i + 1, n):
            factor = A[k][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            B[k] -= factor * B[i]
    X = [0] * n
    for i in range(n - 1, -1, -1):
        X[i] = B[i] - sum(A[i][j] * X[j] for j in range(i + 1, n))
    return X

def regresion_no_lineal(x, y):
    ln_y = [math.log(yi) for yi in y]
    f_ln = regresion_lineal(x, ln_y)
    ln_a = f_ln(0)
    b = f_ln(1) - ln_a
    a = math.exp(ln_a)

    def f(xi):
        return a * math.exp(b * xi)

    return f

def interpolacion_newton(x, y):
    n = len(x)
    coef = y[:]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])

    def f(xi):
        result = coef[-1]
        for i in range(n - 2, -1, -1):
            result = result * (xi - x[i]) + coef[i]
        return result

    return f

def interpolacion_lagrange(x, y):
    def f(xi):
        total = 0
        for i in range(len(x)):
            term = y[i]
            for j in range(len(x)):
                if i != j:
                    term *= (xi - x[j]) / (x[i] - x[j])
            total += term
        return total
    return f

def interpolacion_trazadores(x, y):
    def f(xi):
        for i in range(len(x) - 1):
            if x[i] <= xi <= x[i + 1]:
                m = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                return y[i] + m * (xi - x[i])
        return y[-1]
    return f

def menu():
    metodos = {
        "1": ("Regresión lineal", regresion_lineal),
        "2": ("Regresión polinomial", regresion_polinomial),
        "3": ("Regresión no lineal", regresion_no_lineal),
        "4": ("Interpolación de Newton", interpolacion_newton),
        "5": ("Interpolación de Lagrange", interpolacion_lagrange),
        "6": ("Interpolación con trazadores", interpolacion_trazadores),
    }

    while True:
        print("\n--- MENÚ ---")
        for key, (nombre, _) in metodos.items():
            print(f"{key}. {nombre}")
        print("0. Salir")

        opcion = input("Seleccione una opción: ")
        if opcion == "0":
            break
        elif opcion in metodos:
            nombre, metodo = metodos[opcion]
            if opcion == "2":
                grado = int(input("Grado del polinomio: "))
                f = metodo(x, y, grado)
            else:
                f = metodo(x, y)
            graficar(x, y, f, nombre)
        else:
            print("Opción inválida.")

menu()
