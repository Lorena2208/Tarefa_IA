import pandas as pd
from sklearn.model_selection import train_test_split
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


print(df.head())
print(df.info())
print(df.describe())


df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})


X = df.drop(columns=["ID", "diagnosis"]) 
y = df["diagnosis"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


print("Formato dos dados de treino:", X_train.shape, y_train.shape)
print("Formato dos dados de teste:", X_test.shape, y_test.shape)
