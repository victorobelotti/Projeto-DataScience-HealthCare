import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Definir um estilo visual mais bonito para os gráficos
sns.set_theme(style="whitegrid")

# Carregar os dados limpos
try:
    df_limpo = pd.read_csv('healthcare_data_limpo.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'healthcare_data_limpo.csv' não encontrado.")
    print("Por favor, execute o script '02_limpar_dados.py' primeiro.")
    exit()

print("--- Dados Limpos Carregados com Sucesso ---")
print(df_limpo.head()) # Mostra as 5 primeiras linhas


# --- GRÁFICO 1: Histograma do Tempo de Espera na Emergência ---
# Nosso foco da Pergunta 1 do levantamento de requisitos.

# Filtra apenas quem passou pela emergência (Tempo > 0)
df_emergencia = df_limpo[df_limpo['Tempo_Espera_Emergencia_Min'] > 0]

plt.figure(figsize=(10, 6)) # Define o tamanho da janela do gráfico
sns.histplot(df_emergencia['Tempo_Espera_Emergencia_Min'], bins=30, kde=True)
plt.title('Distribuição do Tempo de Espera na Emergência (Minutos)', fontsize=16)
plt.xlabel('Tempo de Espera (Minutos)')
plt.ylabel('Contagem de Pacientes')

# Salva o gráfico como imagem (para o seu relatório PDF)
plt.savefig('grafico_01_tempo_espera_hist.png')
plt.show() # Mostra o gráfico na tela


# --- GRÁFICO 2: Boxplot do Tempo de Espera por Tipo de Admissão ---
# Isso nos ajuda a ver outliers e comparar os tipos.

plt.figure(figsize=(10, 6))
sns.boxplot(x='Tipo_Admissao', y='Tempo_Espera_Emergencia_Min', data=df_limpo)
plt.title('Tempo de Espera por Tipo de Admissão', fontsize=16)
plt.xlabel('Tipo de Admissão')
plt.ylabel('Tempo de Espera (Minutos)')

# Salva o gráfico
plt.savefig('grafico_02_tempo_espera_boxplot.png')
plt.show()


# --- GRÁFICO 3: Contagem da Nossa Variável Alvo (Readmissão) ---
# Isso é MUITO importante para a etapa de Modelagem.
# Precisamos saber se os dados estão balanceados.

plt.figure(figsize=(8, 5))
sns.countplot(x='Readmissao_30d', data=df_limpo, palette=['#5cb85c', '#d9534f'])
plt.title('Contagem de Pacientes (Readmissão em 30 dias)', fontsize=16)
plt.xlabel('Readmitido? (1 = Sim, 0 = Não)')
plt.ylabel('Número de Pacientes')

# Salva o gráfico
plt.savefig('grafico_03_balanceamento_readmissao.png')
plt.show()

print("--- Análise Exploratória Concluída ---")
print("Gráficos gerados e salvos como arquivos .png na pasta.")