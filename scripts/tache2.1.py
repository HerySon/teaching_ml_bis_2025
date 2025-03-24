import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

# Chargement du fichier CSV (assurez-vous que le chemin du fichier est correct)
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=1000, sep='\t', encoding="utf-8")

# 1. Exploration initiale du dataset
print("Aperçu des premières lignes du dataset :")
print(df.head())
print("\nListe des colonnes :")
print(df.columns.tolist())

# 2. Identification des corrélations entre les variables numériques
# Sélection des colonnes numériques
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Calcul de la matrice de corrélation
corr_matrix = df[numeric_columns].corr()

# Visualisation de la matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matrice de Corrélation")
plt.show()

# Suppression des features fortement corrélées (correlation > 0.9)
# Trouver les colonnes avec une corrélation absolue > 0.9
high_corr_vars = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            colname = corr_matrix.columns[i]
            high_corr_vars.add(colname)

# Supprimer les variables fortement corrélées
df.drop(columns=high_corr_vars, axis=1, inplace=True)
print("\nColonnes après suppression des variables corrélées :")
print(df.columns.tolist())

# 3. Supprimer des colonnes inutiles en fonction de la connaissance du domaine
# Par exemple, supprimer des variables comme 'url', 'creator', 'categories_tags', etc.
columns_to_remove = ['url', 'creator', 'categories_tags', 'created_t', 'created_datetime', 'last_modified_t']
df.drop(columns=columns_to_remove, axis=1, inplace=True)

# Vérification après suppression
print("\nColonnes après suppression manuelle des variables inutiles :")
print(df.columns.tolist())

# 4. Sélection de features par importance des variables
# a. Utilisation de RandomForest pour l'importance des features (en utilisant 'categories' comme cible)
# Remarque : Nous devons sélectionner une cible pour entraîner un modèle de sélection de features. Ici, on peut essayer de prédire 'categories'.
# Pour simplifier, utilisons un ensemble de features numériques pour la sélection.

# Remplacer les valeurs manquantes par la médiane pour entraîner un modèle
df.fillna(df.median(), inplace=True)

# Utilisation d'une colonne cible fictive (par exemple, la première colonne numérique)
X = df.drop(columns='energy_100g')  # X = les features
y = df['energy_100g']  # y = la cible (une colonne numérique)

# Initialiser le modèle RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Affichage de l'importance des features
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

# Affichage des features importantes
print("\nImportance des features avec RandomForest :")
print(importance)

# Sélectionner les features importantes (par exemple, on garde seulement les features ayant une importance > 0.05)
selected_features = importance[importance > 0.05].index.tolist()
df_selected = df[selected_features]

# 5. Sélection de features avec LassoCV (méthode alternative)
# Utilisation de Lasso pour la sélection de features
lasso = LassoCV(cv=5)
lasso.fit(X, y)

# Sélectionner les variables ayant une coefficient non nul
model = SelectFromModel(lasso, threshold="mean", max_features=5)
model.fit(X, y)

selected_features_lasso = X.columns[model.get_support()]
df_selected_lasso = df[selected_features_lasso]

# 6. Résumé et résultats
print("\nColonnes après sélection de features avec RandomForest :")
print(df_selected.columns.tolist())
print("\nColonnes après sélection de features avec LassoCV :")
print(df_selected_lasso.columns.tolist())

# Visualisation des distributions des features sélectionnées
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_selected)
plt.title("Distribution des Features Sélectionnées (RandomForest)")
plt.xticks(rotation=90)
plt.show()

