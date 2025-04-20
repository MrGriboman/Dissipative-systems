import numpy as np
import matplotlib.pyplot as plt

# Функция, описывающая систему ОДУ для модели брюсселятора
def f(t, state, A, B):
    X, Y = state
    dX_dt = A - (B + 1) * X + X**2 * Y
    dY_dt = B * X - X**2 * Y
    return np.array([dX_dt, dY_dt])

# Реализация одного шага метода Рунге-Кутты 4-го порядка
def rk4_step(func, t, state, dt, A, B):
    k1 = func(t, state, A, B)
    k2 = func(t + dt/2, state + dt/2 * k1, A, B)
    k3 = func(t + dt/2, state + dt/2 * k2, A, B)
    k4 = func(t + dt, state + dt * k3, A, B)
    new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state

def solve_brusselator(A, B, t0, t_end, dt, initial_state):
    t_values = np.arange(t0, t_end, dt)
    states = np.empty((len(t_values), 2))
    states[0] = initial_state
    for i in range(1, len(t_values)):
        states[i] = rk4_step(f, t_values[i-1], states[i-1], dt, A, B)
    return t_values, states

def main():
    try:
        A = float(input("Введите значение A: "))
        B = float(input("Введите значение B: "))
    except Exception as e:
        print("Ошибка ввода. Используются значения по умолчанию A=1.0, B=3.0.")
        A, B = 1.0, 3.0

    t0 = 0.0      
    t_end = 400.0  
    dt = 0.01     
    initial_state = np.array([1.0, 1.0]) 

    t_values, states = solve_brusselator(A, B, t0, t_end, dt, initial_state)
    X_values = states[:, 0]
    Y_values = states[:, 1]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_values, X_values, label="X(t)")
    plt.plot(t_values, Y_values, label="Y(t)")
    plt.xlabel("Время")
    plt.ylabel("Концентрация")
    plt.title("Временные ряды X(t) и Y(t)")
    plt.legend()
    
    # Построение фазового портрета (Y от X)
    plt.subplot(1, 2, 2)
    plt.plot(X_values, Y_values)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Фазовый портрет")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
