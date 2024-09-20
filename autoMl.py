# Importar Bibliotecas Necessárias
import h2o
from h2o.automl import H2OAutoML
from google.colab import drive
import pandas as pd

# 1. Inicialização e Configuração
# Inicializar H2O
h2o.init()

# Montar o Google Drive
drive.mount('/content/drive')

# Definir o caminho do arquivo CSV no Google Drive
file_path = "/content/drive/MyDrive/Colab_Notebooks/DENGUE1_SINAN_2021.csv"

# 2. Carregamento e Pré-processamento dos Dados
# Carregar o arquivo CSV em um H2OFrame
df_h2o = h2o.import_file(path=file_path, sep=';')

# Exibir as primeiras linhas do H2OFrame
print("Primeiras linhas do H2OFrame:")
print(df_h2o.head())

# Definir Variáveis de Interesse
col_idade = 'NU_IDADE_N'
col_sexo = 'CS_SEXO'
col_gestante = 'CS_GESTANT'
col_hospitalizado = 'HOSPITALIZ'

# Comorbidades de Interesse
comorbidades = ['DIABETES', 'HEMATOLOG', 'HEPATOPAT', 'RENAL', 'HIPERTENSA', 'ACIDO_PEPT', 'AUTO_IMUNE']

# Garantir que a coluna de hospitalização está no formato binário (1 = hospitalizado, 0 = não hospitalizado)
df_h2o[col_hospitalizado] = (df_h2o[col_hospitalizado] == 1).ifelse(1, 0)

# --- Ajustar a Idade (NU_IDADE_N) ---
# Função para ajustar a idade conforme os critérios
def ajustar_idade_h2o(h2o_frame, coluna_idade):
    idade_ajustada = (
        (h2o_frame[coluna_idade] >= 3000) & (h2o_frame[coluna_idade] < 4000)
    ).ifelse(
        h2o_frame[coluna_idade] - 3000,
        (
            (h2o_frame[coluna_idade] >= 4000) & (h2o_frame[coluna_idade] < 5000)
        ).ifelse(
            h2o_frame[coluna_idade] - 4000,
            -1  # Substituir valores inválidos por -1
        )
    )
    return idade_ajustada

# Aplicar a função para ajustar a idade
idade_ajustada = ajustar_idade_h2o(df_h2o, col_idade)
idade_ajustada.set_names(["Idade_Ajustada"])  # Renomear a coluna

# Adicionar a coluna 'Idade_Ajustada' ao H2OFrame original
df_h2o = df_h2o.cbind(idade_ajustada)

# 3. Análises Descritivas com H2O

# --- 3.1. Taxa de Hospitalização por Faixa Etária ---
print("\n--- Taxa de Hospitalização por Faixa Etária ---")
grouped_idade = df_h2o.group_by("Idade_Ajustada")
# Contar o número total de casos e somar os casos hospitalizados
idade_gravidade = grouped_idade.count().sum(col_hospitalizado)
# Extrair os resultados como um H2OFrame
idade_gravidade_frame = idade_gravidade.get_frame()
# Calcular a taxa de hospitalização
idade_gravidade_frame["Taxa_de_Hospitalizacao (%)"] = (
    idade_gravidade_frame["sum_" + col_hospitalizado] / idade_gravidade_frame["nrow"]
) * 100
# Exibir o resultado
print(idade_gravidade_frame)

# --- 3.2. Taxa de Hospitalização por Sexo ---
print("\n--- Taxa de Hospitalização por Sexo ---")
grouped_sexo = df_h2o.group_by(col_sexo)
sexo_gravidade = grouped_sexo.count().sum(col_hospitalizado)
sexo_gravidade_frame = sexo_gravidade.get_frame()
sexo_gravidade_frame["Taxa_de_Hospitalizacao (%)"] = (
    sexo_gravidade_frame["sum_" + col_hospitalizado] / sexo_gravidade_frame["nrow"]
) * 100
print(sexo_gravidade_frame)

# --- 3.3. Taxa de Hospitalização entre Gestantes ---
print("\n--- Taxa de Hospitalização entre Gestantes ---")
# Filtrar gestantes (CS_GESTANT == 5)
gestantes = df_h2o[df_h2o[col_gestante] == 5]
total_gestantes = gestantes.nrow
# Calcular a soma de hospitalizações
hospitalizadas_gestantes_frame = gestantes[col_hospitalizado].sum()
# Verificar o tipo de retorno e extrair o valor corretamente
if isinstance(hospitalizadas_gestantes_frame, h2o.H2OFrame):
    # Converter para pandas para facilitar o acesso
    hospitalizadas_gestantes = hospitalizadas_gestantes_frame.as_data_frame(use_pandas=True).iloc[0,0]
else:
    # Se já for float
    hospitalizadas_gestantes = hospitalizadas_gestantes_frame

# Calcular a taxa de hospitalização entre gestantes
taxa_hospitalizacao_gestantes = (hospitalizadas_gestantes / total_gestantes) * 100
print(f"Total de Gestantes com Dengue: {total_gestantes}")
print(f"Gestantes Hospitalizadas: {hospitalizadas_gestantes}")
print(f"Taxa de Hospitalização entre Gestantes: {taxa_hospitalizacao_gestantes:.2f}%")

# --- 3.4. Taxa de Hospitalização por Comorbidades ---
print("\n--- Taxa de Hospitalização por Comorbidades ---")
for comorbidade in comorbidades:
    print(f"\nTaxa de Hospitalização por {comorbidade}:")
    grouped_comorb = df_h2o.group_by(comorbidade)
    comorb_gravidade = grouped_comorb.count().sum(col_hospitalizado)
    comorb_gravidade_frame = comorb_gravidade.get_frame()
    comorb_gravidade_frame["Taxa_de_Hospitalizacao (%)"] = (
        comorb_gravidade_frame["sum_" + col_hospitalizado] / comorb_gravidade_frame["nrow"]
    ) * 100
    print(comorb_gravidade_frame)

# --- 3.5. Análise Combinada: Idade, Sexo e Gestação ---
print("\n--- Análise Combinada: Idade, Sexo e Gestação ---")
grouped_combinado = df_h2o.group_by(["Idade_Ajustada", col_sexo, col_gestante])
combinado_gravidade = grouped_combinado.count().sum(col_hospitalizado)
combinado_gravidade_frame = combinado_gravidade.get_frame()
combinado_gravidade_frame["Taxa_de_Hospitalizacao (%)"] = (
    combinado_gravidade_frame["sum_" + col_hospitalizado] / combinado_gravidade_frame["nrow"]
) * 100
print(combinado_gravidade_frame)

# 4. Análises Adicionais com H2O e pandas

# --- 4.1. Casos de Dengue por Mês ---
print("\n--- Casos de Dengue e Taxa de Hospitalização por Mês ---")
# Converter coluna de data para H2O
col_data_notificacao = 'DT_NOTIFIC'
# Converter para data no H2OFrame
df_h2o[col_data_notificacao] = df_h2o[col_data_notificacao].as_date("%d/%m/%Y")
# Extrair o mês da data de notificação
df_h2o['Mes_Notificacao'] = df_h2o[col_data_notificacao].month()
# Agrupar por mês e calcular total de casos e hospitalizações
grouped_mes = df_h2o.group_by("Mes_Notificacao")
casos_por_mes = grouped_mes.count().sum(col_hospitalizado)
casos_por_mes_frame = casos_por_mes.get_frame()
casos_por_mes_frame["Taxa_de_Hospitalizacao (%)"] = (
    casos_por_mes_frame["sum_" + col_hospitalizado] / casos_por_mes_frame["nrow"]
) * 100
print(casos_por_mes_frame)

# --- 4.2. Gravidade dos Casos por Escolaridade e Raça ---
print("\n--- Gravidade dos Casos por Escolaridade e Raça ---")
col_escolaridade = 'CS_ESCOL_N'
col_raca = 'CS_RACA'

grouped_escolar_raca = df_h2o.group_by([col_escolaridade, col_raca])
escolar_raca_gravidade = grouped_escolar_raca.count().sum(col_hospitalizado)
escolar_raca_gravidade_frame = escolar_raca_gravidade.get_frame()
escolar_raca_gravidade_frame["Taxa_de_Hospitalizacao (%)"] = (
    escolar_raca_gravidade_frame["sum_" + col_hospitalizado] / escolar_raca_gravidade_frame["nrow"]
) * 100
print(escolar_raca_gravidade_frame)

# --- 4.3. Correlação entre Sintomas e Hospitalização ---
print("\n--- Correlação entre Sintomas e Hospitalização ---")
# Lista dos sintomas
sintomas = [
    'FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA', 'DOR_COSTAS', 'CONJUNTVIT',
    'ARTRITE', 'ARTRALGIA', 'PETEQUIA_N', 'LEUCOPENIA', 'LACO', 'DOR_RETRO'
]

# Garantir que os sintomas e hospitalização estão no formato binário (1 = presente/hospitalizado, 0 = ausente)
for sintoma in sintomas:
    df_h2o[sintoma] = (df_h2o[sintoma] == 1).ifelse(1, 0)

# Calcular o número de sintomas presentes para cada paciente
# H2O não possui uma função direta para somar colunas, então criaremos uma expressão
df_h2o['NUM_SINTOMAS'] = (
    df_h2o['FEBRE'] + df_h2o['MIALGIA'] + df_h2o['CEFALEIA'] + df_h2o['EXANTEMA'] +
    df_h2o['VOMITO'] + df_h2o['NAUSEA'] + df_h2o['DOR_COSTAS'] + df_h2o['CONJUNTVIT'] +
    df_h2o['ARTRITE'] + df_h2o['ARTRALGIA'] + df_h2o['PETEQUIA_N'] + df_h2o['LEUCOPENIA'] +
    df_h2o['LACO'] + df_h2o['DOR_RETRO']
)

# Limitar o número de sintomas a 14, se necessário (apenas por segurança)

#df_h2o['NUM_SINTOMAS'] = h2o.assign(h2o.pmin(df_h2o['NUM_SINTOMAS'], 14), df_h2o['NUM_SINTOMAS'])

# Converter o H2OFrame para pandas para análises que requerem pandas
df_pandas = df_h2o.as_data_frame(use_pandas=True)

# Calcular a correlação entre cada sintoma e a hospitalização usando pandas
correlacao_sintomas = df_pandas[sintomas + [col_hospitalizado]].corr()[col_hospitalizado].drop(col_hospitalizado).sort_values(ascending=False)
print("Correlação entre sintomas e hospitalização:")
print(correlacao_sintomas)

# --- 4.4. Taxa de Hospitalização por Sintoma ---
print("\n--- Taxa de Hospitalização por Sintoma ---")
# Inicializar um DataFrame para armazenar as taxas# Create an empty list to store results
taxa_hospitalizacao_sintomas_list = []

# Loop to calculate the rates
for sintoma in sintomas:
    total_com_sintoma = df_pandas[df_pandas[sintoma] == 1].shape[0]
    hospitalizados_com_sintoma = df_pandas[(df_pandas[sintoma] == 1) & (df_pandas[col_hospitalizado] == 1)].shape[0]
    taxa = (hospitalizados_com_sintoma / total_com_sintoma) * 100 if total_com_sintoma > 0 else 0

    # Append the result to the list
    taxa_hospitalizacao_sintomas_list.append({
        'Sintoma': sintoma,
        'Pacientes_com_Sintoma': total_com_sintoma,
        'Hospitalizados_com_Sintoma': hospitalizados_com_sintoma,
        'Taxa_de_Hospitalização_%': taxa
    })

# Convert the list to a DataFrame
taxa_hospitalizacao_sintomas = pd.DataFrame(taxa_hospitalizacao_sintomas_list)

# Display the results
print("\nTaxa de hospitalização por sintoma:")
print(taxa_hospitalizacao_sintomas.sort_values(by='Taxa_de_Hospitalização_%', ascending=False))

# --- 4.5. Distribuição por Faixa Etária e Sexo ---
print("\n--- Distribuição por Faixa Etária e Sexo ---")
# Transformar a coluna 'Idade_Ajustada' para anos (já foi ajustada anteriormente)
# Definir faixas etárias
bins = [0, 1, 5, 12, 18, 30, 45, 60, 80, 100]  # Ajustar conforme necessário
labels = ['<1 ano', '1-5', '6-12', '13-18', '19-30', '31-45', '46-60', '61-80', '81+']

# Criar uma nova coluna de faixa etária, tratando valores inválidos (-1) como NaN
df_pandas['IDADE_ANOS'] = df_pandas['Idade_Ajustada'].apply(lambda x: x if x != -1 else None)
df_pandas['FAIXA_ETARIA'] = pd.cut(df_pandas['IDADE_ANOS'], bins=bins, labels=labels, include_lowest=True)

# Contar o número de infectados por sexo e faixa etária
contagem_sexo_faixa = df_pandas.groupby(['FAIXA_ETARIA', col_sexo]).size().unstack(fill_value=0)
print("Contagem por faixa etária e sexo:")
print(contagem_sexo_faixa)

# Calcular a porcentagem de infectados por faixa etária e sexo
contagem_sexo_faixa_pct = contagem_sexo_faixa.div(contagem_sexo_faixa.sum(axis=1), axis=0) * 100
print("\nPorcentagem por faixa etária e sexo:")
print(contagem_sexo_faixa_pct)

# 5. Modelagem Preditiva com H2OAutoML

print("\n--- Iniciando Modelagem Preditiva com H2OAutoML ---")
# Preparar os dados para modelagem
# Definir as variáveis independentes (X) e a variável dependente (y)
# Preparar os dados para modelagem
# Definir as variáveis independentes (X) e a variável dependente (y)
# Preparar os dados para modelagem
# Definir as variáveis independentes (X) e a variável dependente (y)
y = col_hospitalizado  # Prever hospitalização

# Excluir a coluna ACIDO_PEPT da lista de variáveis independentes
# Lista completa de variáveis independentes
X = [
    'NU_NOTIFIC', 'TP_NOT', 'ID_AGRAVO', 'DT_NOTIFIC', 'SEM_NOT', 'NU_ANO', 'SG_UF_NOT', 
    'ID_MUNICIP', 'ID_REGIONA', 'ID_UNIDADE', 'DT_SIN_PRI', 'SEM_PRI', 'NU_IDADE_N', 
    'CS_SEXO', 'CS_GESTANT', 'CS_RACA', 'CS_ESCOL_N', 'ID_CNS_SUS', 'SG_UF', 'ID_MN_RESI', 
    'ID_RG_RESI', 'ID_DISTRIT', 'ID_BAIRRO', 'NM_BAIRRO', 'ID_LOGRADO', 'NM_LOGRADO', 
    'NU_NUMERO', 'NM_COMPLEM', 'ID_GEO1', 'ID_GEO2', 'NM_REFEREN', 'NU_CEP', 'NU_DDD_TEL', 
    'NU_TELEFON', 'CS_ZONA', 'ID_PAIS', 'DT_INVEST', 'ID_OCUPA_N', 'FEBRE', 'MIALGIA', 
    'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA', 'DOR_COSTAS', 'CONJUNTVIT', 'ARTRITE', 
    'ARTRALGIA', 'PETEQUIA_N', 'LEUCOPENIA', 'LACO', 'DOR_RETRO', 'DIABETES', 
    'HEMATOLOG', 'HEPATOPAT', 'RENAL', 'HIPERTENSA', 'ACIDO_PEPT', 'AUTO_IMUNE', 
    'DT_CHIK_S1', 'DT_CHIK_S2', 'DT_PRNT', 'RES_CHIKS1', 'RES_CHIKS2', 'RESUL_PRNT', 
    'DT_SORO', 'RESUL_SORO', 'DT_NS1', 'RESUL_NS1', 'DT_VIRAL', 'RESUL_VI_N', 
    'DT_PCR', 'RESUL_PCR_', 'SOROTIPO', 'HISTOPA_N', 'IMUNOH_N', 'HOSPITALIZ', 
    'DT_INTERNA', 'UF', 'MUNICIPIO', 'HOSPITAL', 'DDD_HOSP', 'TEL_HOSP', 'TPAUTOCTO', 
    'COUFINF', 'COPAISINF', 'COMUNINF', 'CODISINF', 'CO_BAINF', 'NOBAIINF', 
    'CLASSI_FIN', 'CRITERIO', 'DOENCA_TRA', 'CLINC_CHIK', 'EVOLUCAO', 'DT_OBITO', 
    'DT_ENCERRA', 'ALRM_HIPOT', 'ALRM_PLAQ', 'ALRM_VOM', 'ALRM_SANG', 'ALRM_HEMAT', 
    'ALRM_ABDOM', 'ALRM_LETAR', 'ALRM_HEPAT', 'ALRM_LIQ', 'DT_ALRM', 'Idade_Ajustada', 
    'Mes_Notificacao'
]

X.remove('ACIDO_PEPT')  # Remove ACIDO_PEPT, caso esteja na lista

# Garantir que a coluna de hospitalização está no formato categórico
df_h2o[y] = df_h2o[y].asfactor()

# Tratar valores faltantes removendo ou imputando (aqui, remover linhas com Idade_Ajustada == -1)
data = df_h2o[df_h2o['Idade_Ajustada'] != -1, :]

# Dividir os dados em treino e teste
train, test = data.split_frame(ratios=[0.8], seed=42)


# Configurar H2OAutoML para classificação
aml = H2OAutoML(max_models=20, seed=42, sort_metric="AUC", max_runtime_secs=1800)
aml.train(x=X, y=y, training_frame=train)


# Exibir o leaderboard para ver os melhores modelos
lb = aml.leaderboard
print("\n--- Leaderboard H2OAutoML ---")
print(lb)

# Salvar o melhor modelo
best_model = aml.leader
model_path = h2o.save_model(model=best_model, path="/content/drive/MyDrive/Colab_Notebooks", force=True)
print(f"Modelo salvo em: {model_path}")

# Exibir as importâncias das variáveis se o modelo tiver essa propriedade
if hasattr(best_model, 'varimp'):
    print("\nImportância das variáveis:")
    varimp = best_model.varimp(use_pandas=True)
    print(varimp)
else:
    print("\nO modelo selecionado não possui importância das variáveis.")

# Para modelos de regressão logística, exibir os coeficientes
if 'glm' in best_model.algo:
    coef_table = best_model.coef_table()
    print("\nCoeficientes das variáveis:")
    print(coef_table)

# 6. Finalização da Sessão H2O
h2o.shutdown(prompt=False)
