# %%
import os
import shutil
import kagglehub
import polars

# %%
path_origem = kagglehub.dataset_download("franckepeixoto/tabela-fipe")

# %%
diretorio_alvo = "car_datasets"
arquivo_final = os.path.join(diretorio_alvo, "cars_raw.csv")
os.makedirs(diretorio_alvo, exist_ok=True)

# %%
arquivos_baixados = os.listdir(path_origem)

if arquivos_baixados:
    arquivo_origem = os.path.join(path_origem, arquivos_baixados[0])
    try:
        shutil.move(arquivo_origem, arquivo_final)
        print(f"Dataset movido para: {arquivo_final}")
    except Exception:
        print(f"Erro ao mover o arquivo: {Exception}")
else:
    print("Erro: Nenhum arquivo encontrado no download.")

# %%
df = polars.read_csv(arquivo_final)

arquivo_parquet = os.path.join(diretorio_alvo, "cars_raw.parquet")
df.write_parquet(arquivo_parquet)

print(f"Dataset otimizado para Parquet: {arquivo_parquet}")