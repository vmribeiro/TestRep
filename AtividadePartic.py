# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

#
# **************** NA ENTREGA DEVE TER A MATRIZ DE CONFUSAO EM MODO TEXTO (PROCURAR NO
# GOOGLE COMO TEXT CONFUSION MATRIX OU ALGO PARECIDO PARA ARRANJAR O CODIGO) **********
# O RELATORIO É SÓ COMENTAR OS RESULTADOS, DETALHADAMENTE, PRINCIPALMENTE NA MATRIZ DE CONFUSAO
#

#essa parte do codigo é só pra testar a base
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape )

import matplotlib.pyplot as plt
def plot_digits(data,n):
    fig, axes = plt.subplots(n, 10, figsize=(10, n),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
         ax.imshow(data[i].reshape(8, 8),
                   cmap='binary',
                   interpolation='nearest',
                   clim=(0, 16))

plot_digits(digits.data,7)

# cada numero é representado por uma matriz com 64 "quadradinhos
# Existem 1700 e poucos numeros com 64 pontos nessa matriz ou seja, são 64 dimensoes x 1000 e pouco
# o kmeans deve classificar cada um desses pontos (numeros) em 10 valores (0 a 9)

print("execucao do kmeans")
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

# exibe as imagens dos numeros de 0 a 9 abaixo do texto "execucao do kmeans"
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest',
               cmap=plt.cm.binary)
    
## Acuracidade
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]
print(accuracy_score(digits.target, labels))

    