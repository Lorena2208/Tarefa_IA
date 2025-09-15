# Tarefa_IA

# 1. Explique com suas palavras o que significa classificação supervisionada?

Classificação supervisionada é um tipo de aprendizado de máquina em que o modelo aprende a partir de dados já rotulados. Isso significa que o dataset já contém entradas e saídas definidas. O algoritmo usa esses dados para aprender padrões e depois conseguir prever a classe de novos exemplos desconhecidos.


# 2. Qual é a importância de dividir o dataset em conjunto de treino e teste?

A divisão entre treino e teste é importante para saber se o modelo realmente aprendeu ou apenas decorou os dados, avaliando se o modelo vai performar bem em dados novos. O conjunto de treino serve para o algoritmo aprender os padrões, enquanto o de teste é usado para verificar seu desempenho em dados que ele nunca viu. Assim, conseguimos medir a capacidade de generalização do modelo.


# 3. Qual a diferença entre Decision Tree e KNN em termos de funcionamento?

A Decision Tree funciona construindo uma árvore de decisões, onde cada nó representa uma condição sobre uma variável, e no final o modelo classifica o dado em uma classe. Já o KNN (K-Nearest Neighbors) classifica os “vizinhos” mais próximos no espaço de atributos: ele calcula a distância entre os pontos e atribui a classe mais comum entre os K vizinhos mais próximos. Em resumo, a Decision Tree cria regras de decisão, enquanto o KNN se baseia na proximidade entre os exemplos.

# 4.1 – Escolha de Dataset
O dataset escolhido foi o Breast Cancer Wisconsin (Diagnostic), disponível no UCI Machine Learning Repository. O dataset tem uma variável binária clara: se o tumor é maligno (0) ou benigno (1), com 569 amostras, o suficiente para treinar e testar modelos sem ficar pesado mas também não tão pequeno a ponto de não permitir avaliação de desempenho. Ele contém 30 features numéricas (como área, textura, suavidade, compactação, simetria etc.), o que dá bastante espaço para aplicar diferentes modelos e ver como eles se comportam, além de tratar de um problema real de saúde, o que mostra a aplicação de machine learning em algo de impacto social.

