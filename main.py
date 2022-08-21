import pathlib

import matplotlib.pyplot as plt

plt.interactive(True)

import pandas as pd
import seaborn as sns
import os
import re

import tensorflow as tf
from IPython.core.display_functions import display

from tensorflow import keras
from keras import layers
import numpy as np


def func(value):
    return ''.join(value.splitlines())


def padroniza(x):
    print("valor do X: ")
    print(x)
    print("Valor do trains_stats['mean']: ")
    print(trains_stats['mean'])
    print("Valor do trains_stats['std']: ")
    print(trains_stats['std'])

    return (x - trains_stats['mean']) / trains_stats['std']


def adicionaColunas():
    for column in planilha.columns:
        if (type(column) is str) and "Unnamed" not in column:
            column_namesList.append(column)
    display(column_namesList)


class MostraProgresso(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('-', end='')


def plota_historico(history, limita=False):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(10, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Erro Médio Absoluto (MAE) [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Erro de Treinamento')
    plt.legend(loc='best', fontsize=25)
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Erro de Validação')
    plt.legend(loc='best', fontsize=25)
    if limita == True:
        plt.ylim([0, 5])
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Erro Quadrado Médio (MSE) [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Erro de Treinamento')
    plt.legend(loc='best', fontsize=25)
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Erro de Validação')
    plt.legend(loc='best', fontsize=25)
    if limita == True:
        plt.ylim([0, 20])
    plt.legend()
    plt.show()
    plt.pause(1000)


def controi_modelo():
    # modelo sequencial, com camadas densamente conectadas
    # Função de ativação ReLU (boa para evitar o decaimento excessivo dos pesos e do gradiante)
    # output de um valor numérico, na ultima camada
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    # escolha do otimizador
    optimizer = tf.keras.optimizers.RMSprop(0.01)
    # Função de custo Erro Quadrado Médio (Mean Squared Error)
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse']
    )
    return model


column_namesList = []
planilha = pd.read_excel("Dados tratados.xlsx", sheet_name="Planilha1")
dataset = planilha.copy()
# Adiciona Colunas do arquivoExcel.
adicionaColunas()
# Remove os dados de quando a máquina está parada
dataset.drop(dataset[dataset['O2 (%)'] == 'Parado'].index, inplace=True)
# Verifica quais colunas possuem valores que não sejam numéricos
print("============ Validando se todos os valores na planilha são numéricos e válidos ============")
display(dataset.isna())

print("============ Valores estastíticos para todas as colunas válidas  ============")
display(dataset.describe())

# Separação de variáveis categóricas de variáveis numéricas (contínuas) Obs. Não aplicável para os dados atuais,
# mas é bom para fins de demonstração que nos dados atuais não podemos usar valores categóricos

print("============ Demonstra a quantidade de variáveis únicas e suas variações numéricas ============")
for col in dataset.columns:
    print(
        f"col {col}, Quantidade valores únicos: {len(dataset[col].unique())} , {dataset[col].unique()}  \n"
    )

# Separação de dados para treino e treinamento da rede neural

# Porcentagem do uso de treinamento:  80%
# Procentagem do uso de testes : 20%
pct_split = 0.8
# Separa os dados de teste com os dados de treinamento
train_dataset = dataset.sample(frac=pct_split, random_state=0)
teste_dataset = dataset.drop(train_dataset.index)

# Pair-plots entre os dados contíonuos
# Estudar e aplicar melhor os Pair-plots
# Adiciona os PairPlot para pastas e imagens para melhor dekmonstração posterior
for item in column_namesList:
    path = item.strip() + "_PairPlots"
    path = re.sub('[^A-Za-z\d]+', '', path)
    FileExist = False
    try:
        os.mkdir(path)
    except FileExistsError:
        FileExist = True
    for item2 in column_namesList:
        if item != item2 and FileExist == False:
            sns.pairplot(train_dataset[[item, item2]], diag_kind='kde')
            item = item.strip()
            item2 = item2.strip()
            file_name = item + "_" + item2 + ".png"
            file_name = file_name.replace("/", "")
            file_name = path + "/" + func(file_name)
            print("Salvando o arquivo: " + file_name)
            plt.savefig(file_name)

# Remoção da variavel alvo

include = ['float', 'int']
train_dataset = train_dataset.describe(include=include)
teste_dataset = teste_dataset.describe(include=include)
trains_stats = train_dataset.describe(include=include)
trains_stats.pop('NOx (mg/Nm³)\n10% O2\n650 (mg/Nm³)')
trains_stats = trains_stats.T
print(trains_stats)
train_labels = train_dataset.pop('NOx (mg/Nm³)\n10% O2\n650 (mg/Nm³)')
test_labels = teste_dataset.pop('NOx (mg/Nm³)\n10% O2\n650 (mg/Nm³)')

# Normalização dos dados
normed_train_data = padroniza(train_dataset)
normed_test_data = padroniza(teste_dataset)

display(normed_train_data)

# CRIAÇÃO DO MODELO

print(" ===== Criando os modelos =====")

# Utilizando modelo Sequential com camadas densamente conectadas, e a camada de saída deve retornar um valor contínuo
# (não discreto)


model = controi_modelo()
print("Todos os atributos do modelo de deep learning")
print(model.summary())
# Verificar 10 dados de treinamento
example_batch = normed_train_data[:10]
print("Example_batch: ")
print(example_batch)
example_result = model.predict(example_batch)
X = np.asarray(example_result).astype(np.float32)
print(example_result)
EPOCHS = 1000

history = model.fit(
    x=normed_train_data,
    y=train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[MostraProgresso()]
)

print(history.__dict__.keys())
print(history.history.keys())

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.describe())
plota_historico(history)

# Previsão de test

prev_test = model.predict(normed_test_data).flatten()
plt.scatter(test_labels, prev_test)
plt.xlabel('Real', fontsize=18)
plt.ylabel('Previsto', fontsize=18)
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-1000000, 1000000], [-1000000, 1000000])
plt.show()
plt.pause(1000)

# Verificando a distribuição dos erros, e ver se são de distribuição aproximadamente Gaussiana

erro = prev_test - test_labels
plt.hist(erro, bins=25)
plt.xlabel('Prev Erro')
_ = plt.ylabel('Contagem')
sns.displot(a=erro, bins=25)
plt.show()
plt.pause(1000)


