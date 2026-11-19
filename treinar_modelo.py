import csv
import os
import pickle
import numpy as np


K = 5


pasta = os.path.dirname(
    os.path.abspath(__file__)
)

arquivo_dataset = os.path.join(
    pasta,
    "dataset.csv"
)

arquivo_modelo = os.path.join(
    pasta,
    "modelo_selos.pkl"
)


dados = []

with open(
    arquivo_dataset,
    "r",
    encoding="utf-8"
) as arquivo:
    leitor = csv.reader(arquivo)

    next(leitor)

    for linha in leitor:
        dados.append(linha)


X = np.array(
    [linha[:-1] for linha in dados],
    dtype=float
)

y = np.array(
    [linha[-1] for linha in dados]
)


indices = np.arange(
    len(X)
)

np.random.seed(42)
np.random.shuffle(indices)


X = X[indices]
y = y[indices]


quantidade_treino = int(
    len(X) * 0.8
)

X_treino = X[:quantidade_treino]
y_treino = y[:quantidade_treino]

X_teste = X[quantidade_treino:]
y_teste = y[quantidade_treino:]


def prever(amostra):
    diferencas = X_treino - amostra

    distancias = np.sqrt(
        np.sum(
            diferencas ** 2,
            axis=1
        )
    )

    indices_vizinhos = np.argsort(
        distancias
    )[:K]

    classes_vizinhas = y_treino[
        indices_vizinhos
    ]

    classes, quantidades = np.unique(
        classes_vizinhas,
        return_counts=True
    )

    indice_maior = np.argmax(
        quantidades
    )

    return classes[indice_maior]


print()
print("=" * 50)
print("TESTANDO O MODELO KNN")
print("=" * 50)
print()

acertos = 0

for i in range(len(X_teste)):
    previsao = prever(
        X_teste[i]
    )

    if previsao == y_teste[i]:
        acertos += 1


precisao = (
    acertos /
    len(X_teste)
) * 100


print(
    f"Amostras totais: {len(X)}"
)

print(
    f"Amostras para treino: {len(X_treino)}"
)

print(
    f"Amostras para teste: {len(X_teste)}"
)

print(
    f"Quantidade de caracteristicas: {X.shape[1]}"
)

print()

print(
    f"Acertos: {acertos}/{len(X_teste)}"
)

print(
    f"Precisao: {precisao:.2f}%"
)


modelo = {
    "X": X_treino,
    "y": y_treino,
    "k": K
}


with open(
    arquivo_modelo,
    "wb"
) as arquivo:
    pickle.dump(
        modelo,
        arquivo
    )


print()
print("=" * 50)
print("MODELO SALVO!")
print("=" * 50)
print()
print(
    f"Arquivo: {arquivo_modelo}"
)