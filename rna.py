# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# %%
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# %%
modelo_mlp = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='relu', # Infelizmente não temos "leaky ReLU" no scikit-learn
    solver='adam',
    max_iter=2000,
    learning_rate_init=0.01,
)

# %%
modelo_mlp.fit(X, y)

# %%
previsoes = modelo_mlp.predict(X)
score = modelo_mlp.score(X, y)

print(f"Entradas:\n{X}")
print(f"Previsões: {previsoes}")
print(f"Acurácia Final: {score * 100}%")

# %%
plt.figure(figsize=(12, 6))
plt.plot(modelo_mlp.loss_curve_)
plt.title("Convergência da Loss - Scikit-Learn MLP")
print(f"Iterações totais: {modelo_mlp.n_iter_}")
plt.xlabel("Iterações")
plt.ylabel("Log-Loss")
plt.grid(True)
plt.show()