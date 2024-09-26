import numpy as np
import matplotlib.pyplot as plt

# Definindo a função sinc(x)
def sinc(x):
    return np.sin(x) / x

# Criando o intervalo de x de 0.1 até 8
x = np.linspace(0.1, 8, 500)
y = sinc(x)

# Plotando a função
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='sinc(x)')
plt.axhline(0, color='black',linewidth=0.5)
plt.title("Função sinc(x)")
plt.xlabel("x")
plt.ylabel("sinc(x)")
plt.grid(True)
plt.legend()
plt.show()

# Método da Bisseção
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

# Executando o método da Bisseção
a, b = 3.0, 4.0
raiz_bissecao, iteracoes_bissecao, erros_bissecao = bissecao(sinc, a, b)

raiz_bissecao, iteracoes_bissecao[-1], erros_bissecao[-1]  # Retornar o zero encontrado e o erro final
