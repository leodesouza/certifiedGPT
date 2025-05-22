import pandas as pd

# Carrega o log (você pode adaptar o caminho se for um CSV, por exemplo)
df = pd.read_csv("seu_arquivo_log.csv", sep="\t")  # ou sep=',' se for CSV

# Garante que os tipos estejam corretos
df["correct"] = df["correct"].astype(bool)
df["radius"] = pd.to_numeric(df["radius"], errors="coerce")  # trata "-1" como NaN

# Total de exemplos
total = len(df)

# 1. Acurácia padrão (apenas se está correto)
accuracy = df["correct"].sum() / total

# 2. Acurácia certificada (apenas se o raio > 0, mesmo que esteja errado)
certified = (df["radius"] > 0).sum() / total

# 3. Certified Top-1 Accuracy (raio > 0 E resposta correta)
certified_top1 = ((df["radius"] > 0) & (df["correct"] == True)).sum() / total

# Resultados
print(f"Acurácia padrão: {accuracy:.4f}")
print(f"Acurácia certificada (robustez): {certified:.4f}")
print(f"Certified Top-1 Accuracy (robustez + correta): {certified_top1:.4f}")
