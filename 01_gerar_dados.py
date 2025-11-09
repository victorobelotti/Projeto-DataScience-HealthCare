import pandas as pd
import numpy as np

# Definir o número de registros
num_registros = 1000

# Criar dados base
data = {
    'ID_Paciente': np.random.randint(10000, 50000, size=num_registros),
    'ID_Visita': np.arange(1, num_registros + 1),
    'Idade': np.random.randint(18, 90, size=num_registros),
    'Sexo': np.random.choice(['Masculino', 'F', 'Feminino', 'M', 'M'], size=num_registros),
    'Tipo_Admissao': np.random.choice(['Emergência', 'Eletiva', 'Urgência'], size=num_registros, p=[0.5, 0.3, 0.2]),
    'Tempo_Espera_Emergencia_Min': np.nan, # Será preenchido depois
    'Diagnostico_Principal_CID': np.random.choice(['I10', 'E11', 'J45', 'N18', 'I21', 'C50'], size=num_registros),
    'Indice_Comorbidade': np.random.randint(0, 5, size=num_registros),
    'Tempo_Permanencia_Dias': np.random.randint(1, 30, size=num_registros),
    'Nota_Satisfacao': np.random.choice([1, 2, 3, 4, 5, np.nan], size=num_registros, p=[0.05, 0.05, 0.1, 0.3, 0.4, 0.1]),
    'Glicemia_Jejum': np.random.uniform(70, 250, size=num_registros),
    'Readmissao_30d': np.random.choice([0, 1], size=num_registros, p=[0.8, 0.2]) # Variável Alvo
}

df = pd.DataFrame(data)

# --- Introduzindo "sujeira" realista ---

# 1. Preencher Tempo_Espera_Emergencia (só para quem é da 'Emergência')
mask_emergencia = df['Tipo_Admissao'] == 'Emergência'
df.loc[mask_emergencia, 'Tempo_Espera_Emergencia_Min'] = np.random.randint(10, 300, size=mask_emergencia.sum())

# 2. Introduzir mais valores ausentes (NaN)
df.loc[df.sample(frac=0.05).index, 'Glicemia_Jejum'] = np.nan # 5% ausente
df.loc[df.sample(frac=0.03).index, 'Indice_Comorbidade'] = np.nan # 3% ausente

# 3. Introduzir dados duplicados
duplicatas = df.sample(n=15)
df = pd.concat([df, duplicatas]).reset_index(drop=True)

# Salvar no CSV
df.to_csv('healthcare_data_simulado.csv', index=False)

print(f"Arquivo 'healthcare_data_simulado.csv' criado com {len(df)} linhas (incluindo duplicatas).")
print("Dados 'sujos' prontos para limpeza!")