import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

# Ignorar avisos futuros para manter a saída limpa
warnings.filterwarnings('ignore')

print("--- Iniciando Script de Modelagem Preditiva ---")

# Carregar os dados limpos
try:
    df = pd.read_csv('healthcare_data_limpo.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'healthcare_data_limpo.csv' não encontrado.")
    print("Por favor, execute o script '02_limpar_dados.py' primeiro.")
    exit()

# --- 1. Preparação dos Dados (Pré-processamento para ML) ---

# A IA não entende texto (ex: 'Masculino', 'I10'). Precisamos transformar tudo em números.

# Separar os dados em features (X) e alvo (y)
# y é o que queremos prever: 'Readmissao_30d'
y = df['Readmissao_30d']
# X é todo o resto que usamos para prever
X = df.drop(['Readmissao_30d', 'ID_Paciente', 'ID_Visita'], axis=1) # Removemos IDs

# Identificar quais colunas são numéricas e quais são categóricas
colunas_numericas = ['Idade', 'Tempo_Espera_Emergencia_Min', 'Indice_Comorbidade', 
                     'Tempo_Permanencia_Dias', 'Nota_Satisfacao', 'Glicemia_Jejum']
colunas_categoricas = ['Sexo', 'Tipo_Admissao', 'Diagnostico_Principal_CID']

# --- 2. Criação do Pipeline de Pré-processamento ---

# Criamos "transformadores" para automatizar a preparação dos dados

# Para dados numéricos: Vamos padronizar a escala (ex: Idade de 0-90 e Glicemia de 70-250)
transformador_numerico = StandardScaler()

# Para dados categóricos: Vamos usar "One-Hot Encoding"
# (Ex: 'Tipo_Admissao' vira 'Tipo_Admissao_Emergência', 'Tipo_Admissao_Eletiva')
transformador_categorico = OneHotEncoder(handle_unknown='ignore')

# Juntar os transformadores em um "preparador" de colunas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', transformador_numerico, colunas_numericas),
        ('cat', transformador_categorico, colunas_categoricas)
    ])

# --- 3. Criação do Modelo (Regressão Logística) ---
# Usaremos a Regressão Logística, como sugerido no desafio 
modelo = LogisticRegression(max_iter=1000)

# Criar o Pipeline final:
# 1. Prepara os dados (com o 'preprocessor')
# 2. Treina o modelo (com o 'modelo')
pipeline_final = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', modelo)])

# --- 4. Treinamento e Teste do Modelo ---

# Dividir os dados: 80% para treinar, 20% para testar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

# Treinar o modelo (a "IA" aprende com os dados de treino)
print("Treinando o modelo...")
pipeline_final.fit(X_train, y_train)
print("Modelo treinado com sucesso!")

# Fazer previsões (Usar o modelo treinado nos dados de teste)
y_pred = pipeline_final.predict(X_test)

# --- 5. Avaliação dos Resultados ---
# O quão bom é o nosso modelo?

# Acurácia: % de previsões corretas
acuracia = accuracy_score(y_test, y_pred)
print("\n--- RESULTADOS DA AVALIAÇÃO DO MODELO ---")
print(f"Acurácia: {acuracia * 100:.2f}%")

# Relatório de Classificação: Mostra precisão e recall
print("\nRelatório de Classificação:")
# 0 = Não Readmitido, 1 = Readmitido
print(classification_report(y_test, y_pred, target_names=['0 - Não Readmitido', '1 - Readmitido']))

# Matriz de Confusão: Mostra os erros e acertos
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("---------------------------------------------")
print("(Linhas = Real, Colunas = Previsão)")