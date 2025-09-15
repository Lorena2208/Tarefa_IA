import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

colunas = [
    "ID", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

df = pd.read_csv(url, header=None, names=colunas)

print("Primeiras linhas do dataset:")
print(df.head())
print("\nInformações gerais do dataset:")
print(df.info())
print("\nEstatísticas descritivas das features:")
print(df.describe())

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop(columns=["ID", "diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nFormato dos dados de treino:", X_train.shape, y_train.shape)
print("Formato dos dados de teste:", X_test.shape, y_test.shape)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=5000)
}

results = {}

print("\n=====Resultados dos Modelos=====")
for name, model in models.items():
    print(f"\nModelo: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"Acurácia: {acc:.4f}")
    print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred))


print("\n=====Gráfico de Comparação=====")
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=['#4CAF50', '#2196F3', '#FFC107'])
plt.ylabel("Acurácia")
plt.title("Comparação de Acurácia dos Modelos de ML")
plt.ylim(0.85, 1.0) 
plt.grid(axis='y', linestyle='--', alpha=0.7)


for i, v in enumerate(results.values()):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()