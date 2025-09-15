# Projeto de Classificação Supervisionada – Breast Cancer Wisconsin (Diagnostic)

## 1. Escolha do Dataset  
- **Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
- **Motivo da escolha:**  
  - É um dataset real da área da saúde, com relevância prática.  
  - Possui 569 amostras.  
  - O dataset tem uma variável binária clara: se o tumor é maligno (0) ou benigno (1).  
  - Contém 30 features numéricas, que permitem aplicar e comparar diferentes algoritmos de classificação.

---

## 2. Preparação dos Dados  
- **Leitura:** O dataset foi carregado diretamente do repositório UCI no formato .data.  
- **Ajustes realizados:**  
  - Inclusão dos nomes das colunas (conforme documentação oficial).  
  - Conversão da variável alvo diagnosis: M → 1 (maligno), B → 0 (benigno).  
  - Exclusão da coluna ID (não traz informação relevante para a classificação).  
- **Divisão dos dados:**  
  - Conjunto de treino: 70%  
  - Conjunto de teste: 30%  
  - random_state=42 para reprodutibilidade.  

---

## 3. Modelagem  
- Três modelos de classificação supervisionada foram aplicados:  
  - **Decision Tree** (árvore de decisão)  
  - **KNN** (K-Nearest Neighbors)  
  - **Logistic Regression**  
- Todos os modelos foram treinados no conjunto de treino e avaliados no conjunto de teste.  

---

## 4. Avaliação  
- **Métricas utilizadas:**  
  - **Acurácia**: porcentagem de acertos do modelo.  
  - **Matriz de confusão**: análise detalhada de verdadeiros positivos/negativos e falsos positivos/negativos.  
  - **Relatório de classificação**: precision, recall e f1-score.  
- **Resultados obtidos:**  
  - Decision Tree → Acurácia: 0.9415  
  - KNN → Acurácia: 0.9591  
  - Logistic Regression → Acurácia: 0.9766 (melhor desempenho).  

---

## 5. Conclusão  
- O Logistic Regression foi o modelo com melhor desempenho, alcançando 97,66% de acurácia.  
- Ele apresentou bom equilíbrio entre precision e recall, sendo confiável para este tipo de problema.  
- Possíveis próximos passos:  
  - Testar técnicas de normalização ou padronização de dados (especialmente para o KNN).  
  - Ajustar hiperparâmetros dos modelos (como profundidade da árvore ou valor de K).  
  - Explorar outros algoritmos, como SVM ou Random Forest, para comparar os resultados.  
