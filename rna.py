# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# %%
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
alvos = np.array([0, 1, 1, 0])

# %%
modelo_mlp = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='relu',
    solver='adam',
    max_iter=2000,
    learning_rate_init=0.01,
    random_state=42
)

# %%
modelo_mlp.fit(entradas, alvos)

# %%
previsoes = modelo_mlp.predict(entradas)
score = modelo_mlp.score(entradas, alvos)

print(f"Entradas:\n{entradas}")
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
