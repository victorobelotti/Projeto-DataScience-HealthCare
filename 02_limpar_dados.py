import pandas as pd
import numpy as np

# Carregar os dados "sujos"
df = pd.read_csv('healthcare_data_simulado.csv')

print(f"--- ANÁLISE INICIAL (DADOS SUJOS) ---")
print(f"Total de linhas: {len(df)}")
print("\nValores Ausentes (Antes):")
print(df.isnull().sum())
print(f"\nLinhas Duplicadas (Antes): {df.duplicated().sum()}")
print(f"\nValores únicos em 'Sexo' (Antes): {df['Sexo'].unique()}")

# --- ETAPA 1: TRATAR DADOS DUPLICADOS [cite: 56] ---
# Remove linhas que são exatamente iguais
df = df.drop_duplicates()
print(f"\n--- LIMPEZA REALIZADA ---")
print(f"Linhas Duplicadas (Depois): {df.duplicated().sum()}")

# --- ETAPA 2: TRATAR VALORES AUSENTES (Missing Values) [cite: 61] ---
# Estratégia 1: Para 'Tempo_Espera_Emergencia_Min', preencher com 0 onde não for emergência
# (Decisão baseada no domínio: se não foi emergência, o tempo de espera foi 0)
df['Tempo_Espera_Emergencia_Min'] = df['Tempo_Espera_Emergencia_Min'].fillna(0)

# Estratégia 2: Para 'Nota_Satisfacao' e 'Indice_Comorbidade', usar a Mediana (pois são números inteiros/ordinais)
mediana_satisfacao = df['Nota_Satisfacao'].median()
df['Nota_Satisfacao'] = df['Nota_Satisfacao'].fillna(mediana_satisfacao)

mediana_comorbidade = df['Indice_Comorbidade'].median()
df['Indice_Comorbidade'] = df['Indice_Comorbidade'].fillna(mediana_comorbidade)

# Estratégia 3: Para 'Glicemia_Jejum', usar a Média (pois é uma variável contínua)
media_glicemia = df['Glicemia_Jejum'].mean()
df['Glicemia_Jejum'] = df['Glicemia_Jejum'].fillna(media_glicemia)

print("\nValores Ausentes (Depois):")
print(df.isnull().sum()) # Verificar se todos os nulos foram tratados

# --- ETAPA 3: PADRONIZAÇÃO DE VARIÁVEIS [cite: 61] ---
# Padronizar a coluna 'Sexo' para 'M' (Masculino) e 'F' (Feminino)
mapa_sexo = {
    'Masculino': 'M',
    'M': 'M',
    'Feminino': 'F',
    'F': 'F'
}
df['Sexo'] = df['Sexo'].map(mapa_sexo)
print(f"\nValores únicos em 'Sexo' (Depois): {df['Sexo'].unique()}")

# Salvar o DataFrame limpo em um novo CSV (opcional, mas recomendado)
df.to_csv('healthcare_data_limpo.csv', index=False)

print(f"\n--- LIMPEZA CONCLUÍDA ---")
print(f"Dados limpos e salvos em 'healthcare_data_limpo.csv' com {len(df)} linhas.")
print(df.info())