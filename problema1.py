import numpy as np
import matplotlib.pyplot as plt

def sinc(x):
    return np.sin(x) / x

def sinc_deriv(x):
    return (x * np.cos(x) - np.sin(x)) / (x ** 2)

def plot_sinc(x, y):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='sinc(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title("Função sinc(x)")
    plt.xlabel("x")
    plt.ylabel("sinc(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

def bissecao(f, a, b, tol=1e-6, max_iter=100):
    iteracoes = []
    erros = []
    if f(a) * f(b) > 0:
        raise ValueError("f(a) e f(b) devem ter sinais opostos")
    
    for i in range(max_iter):
        c = (a + b) / 2.0
        iteracoes.append(c)
        erros.append(abs(f(c)))
        if abs(f(c)) < tol or (b - a) / 2.0 < tol:
            return c, iteracoes, erros
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c, iteracoes, erros

def falsa_posição(f, a, b, tol=1e-6, max_iter=100):
    iteracoes = []
    erros = []
    if f(a) * f(b) > 0:
        raise ValueError("f(a) e f(b) devem ter sinais opostos")
    
    for i in range(max_iter):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        iteracoes.append(c)
        erros.append(abs(f(c)))
        if abs(f(c)) < tol or (b - a) / 2.0 < tol:
            return c, iteracoes, erros
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c, iteracoes, erros

def newton(f, df, x0, tol=1e-6, max_iter=100):
    iteracoes = []
    erros = []
    x = x0
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < tol:  # Evitar divisão por zero
            raise ValueError("Derivada muito próxima de zero")
        
        iteracoes.append(x)
        erros.append(abs(fx))
        
        if abs(fx) < tol:
            return x, iteracoes, erros
        
        x = x - fx / dfx
    
    return x, iteracoes, erros

def mostrar_resultados(raiz, iteracoes, erros):
    print(f"Raiz: {raiz}")
    print(f"Iterações: {len(iteracoes)}")
    print(f"Erros: {erros[-1]}")

def main():
    x = np.linspace(0.1, 8, 500)
    y = sinc(x)
    
    plot_sinc(x, y)
    
    a, b = 3.0, 4.0

    print("\nMétodo da bisseção de sinc(x) no intervalo" f"[{a}, {b}]")
    raiz_bissecao, iteracoes_bissecao, erros_bissecao = bissecao(sinc, a, b)
    mostrar_resultados(raiz_bissecao, iteracoes_bissecao, erros_bissecao)

    print("\nMétodo da falsa posição de sinc(x) no intervalo" f"[{a}, {b}]")
    raiz_falsa_posicao, iteracoes_falsa_posicao, erros_falsa_posicao = falsa_posição(sinc, a, b)
    mostrar_resultados(raiz_falsa_posicao, iteracoes_falsa_posicao, erros_falsa_posicao)

    x0 = 3.5  # Chute inicial
    print("\nMétodo de Newton-Raphson de sinc(x) com chute inicial" f" x0 = {x0}")
    try:
        raiz_newton, iteracoes_newton, erros_newton = newton(sinc, sinc_deriv, x0)
        mostrar_resultados(raiz_newton, iteracoes_newton, erros_newton)
    except ValueError as e:
        print(f"Erro no método de Newton: {e}")

if __name__ == "__main__":
    main()