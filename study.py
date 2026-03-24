import numpy as np
import statistics as st
import matplotlib.pyplot as plt

random_array = np.random.randint(1, 100, size=20)

media_aritmetica = np.mean(random_array)
# media_geometrica = st.geometric_mean(random_array)
media_harmonica = st.harmonic_mean(random_array)

desvio_padrao_media_aritmetica = np.std(random_array) # Por padrão, o numpy calcula o desvio padrão usando a média aritmética
variancia = [(x - media_harmonica) ** 2 for x in random_array]
desvio_padrao_media_harmonica = np.sqrt(np.mean(variancia))

print(f"\n{", ".join(map(str, random_array))}\n")
print(f"Média aritmética: {media_aritmetica:.2f}")
# print(f"Média geométrica: {media_geometrica:.2f}")
print(f"Média harmônica: {media_harmonica:.2f}")
print(f"\nDesvio padrão: {desvio_padrao_media_aritmetica:.2f}")
print(f"Desvio padrão usando média harmônica: {desvio_padrao_media_harmonica:.2f}")

plt.figure(figsize=(12, 10))
plt.hist(random_array, bins=15, edgecolor="black", alpha=0.7, color="lightgray")

plt.axvline(media_aritmetica, color="blue", linestyle="dashed", linewidth=2, label=f"Média Aritmética: {media_aritmetica:.2f}")
plt.axvspan(media_aritmetica - desvio_padrao_media_aritmetica, media_aritmetica + desvio_padrao_media_aritmetica, color="blue", alpha=0.2, label="1 Desvio (Aritmética)")

plt.axvline(media_harmonica, color="red", linestyle="dashed", linewidth=2, label=f"Média Harmônica: {media_harmonica:.2f}")
plt.axvspan(media_harmonica - desvio_padrao_media_harmonica, media_harmonica + desvio_padrao_media_harmonica, color="red", alpha=0.2, label="1 Desvio (Harmônica)")

plt.title("Histograma e Áreas de Dispersão (Desvio Padrão)")
plt.xlabel("Valores")
plt.ylabel("Frequência")
plt.legend()

plt.show()

"""
Comprovando que média harmônica não combina com desvio padrão,
para isso teríamos que usar outras técnicas (e como são mais de uma, não irei citá-las aqui)
"""