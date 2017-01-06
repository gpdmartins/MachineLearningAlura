import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier


modelo = MultinomialNB()
df = pd.read_csv('buscas.csv')
X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']
Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df
X = Xdummies_df.values
Y = Ydummies_df.values


porcentagem_treino = 0.8
porcentagem_teste = 0.1

tamanho_treino = len(Y)*porcentagem_treino
tamanho_teste = len(Y) * porcentagem_teste
tamanho_validacao = len(Y) - tamanho_treino - tamanho_teste

fim_de_teste = tamanho_treino + tamanho_teste

treino_dados = X[:int(tamanho_treino)]
treino_marcacoes = Y[:int(tamanho_treino)]

teste_dados = X[int(tamanho_treino):int(fim_de_teste)]
teste_marcacoes = Y[int(tamanho_treino):int(fim_de_teste)]

validacao_dados = X[int(fim_de_teste):]
validacao_marcacoes = Y[int(fim_de_teste):]


#aula 5
def fit_and_predict(nome, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)
    resultado = modelo.predict(teste_dados)
    acertos = resultado == teste_marcacoes
    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    msg = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto



modelo = MultinomialNB()
resultado_Multinomial = fit_and_predict("MultinomialNB", treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

modelo = AdaBoostClassifier()
resultado_AdaBoost = fit_and_predict("AdaBoostClassifier", treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if resultado_Multinomial > resultado_AdaBoost:
    vencedor = resultado_Multinomial
else:
    vencedor = resultado_AdaBoost

def teste_real(vencedor, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
    print(msg)

teste_real(vencedor, validacao_dados, validacao_marcacoes)

#aula 4
acerto_base = max(Counter(validacao_marcacoes).itervalues())
taxa_de_acerto_validacao = 100.0*acerto_base/len(validacao_marcacoes)
print("Taxa de acerto base: %.1f%%" %(taxa_de_acerto_validacao))

total_de_elementos = len(teste_dados)
print("Total de teste: %d" %total_de_elementos)
