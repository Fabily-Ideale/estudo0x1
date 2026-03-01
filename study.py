import numpy as np
import statistics as st

random_array = np.random.randint(1, 21, size=10)

media_aritmetica = np.mean(random_array)
media_geometrica = st.geometric_mean(random_array)
media_harmonica = st.harmonic_mean(random_array)

desvio_padrao_media_aritmetica = np.std(random_array) #Por padrão, o numpy calcula o desvio padrão usando a média aritmética
variancia = [(x - media_harmonica) ** 2 for x in random_array]
desvio_padrao_media_harmonica = np.sqrt(np.mean(variancia))

print(f"\n{", ".join(map(str, random_array))}\n")
print(f"Médio aritmética: {media_aritmetica:.2f}")
print(f"Média geométrica: {media_geometrica:.2f}")
print(f"Média harmônica: {media_harmonica:.2f}")
print(f"Desvio padrão amostral: {dp_a:.2f}")
print(f"Desvio padrão populacional: {dp_p:.2f}")