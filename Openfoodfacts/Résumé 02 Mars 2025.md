## Chargement des Données

- **Source** : Fichier `en.openfoodfacts.org.products.csv`
- **Taille** : 300,000 produits alimentaires
- **Colonnes** : 206 colonnes initialement
- **Données manquantes** : 196 colonnes contiennent des valeurs manquantes
- **Doublons** : Aucune ligne dupliquée

---

## 2️⃣ Nettoyage et Sélection des Variables

### 📌 **Colonnes Conservées** (18 au total) :

✅ **16 variables numériques**  
✅ **1 variable ordinale** : `nutriscore_grade`  
✅ **1 variable nominale** : `pnns_groups_1`

### 🗑 **Colonnes Supprimées** (188) :

❌ **Trop de valeurs manquantes** (>50%)  
❌ **Trop de catégories uniques**  
❌ **Variance trop faible**  
❌ **Corrélations trop élevées**

### 🔗 **Corrélations Identifiées** :

- **Corrélation parfaite (1.00)** entre :
    - `energy-kcal_100g` et `energy_100g` (différentes unités)
    - `salt_100g` et `sodium_100g` (différentes unités)

---

## 3️⃣ Analyse de la Qualité des Données

### 🥧 **Répartition des Types de Variables** (Graphique en camembert)

📌 Permet de visualiser la présence de :

- `float64` : Nombres décimaux
- `int64` : Nombres entiers
- `object` : Texte
- `category` : Catégories ordonnées et non ordonnées

### 🔥 **Matrice de Corrélation** (Heatmap)

📌 Indique les relations entre variables numériques :

- **Rouge** : Corrélation positive forte (~1)
- **Bleu** : Corrélation négative forte (~-1)
- **Blanc** : Absence de corrélation (~0)

✏️ Exemple : Une forte corrélation entre `energy_100g` et `energy-kcal_100g` est logique car elles mesurent la même chose.

### 📊 **Distribution des Variables Numériques** (Boxplots)

📌 Regroupement par 5 variables maximum pour plus de lisibilité.  
📌 Chaque boîte montre :

- **Médiane** (ligne rouge)
- **50% des données** (boîte)
- **Valeurs aberrantes** (points)  
    📌 Statistiques détaillées affichées :
- Moyenne
- Écart-type
- Valeurs min/max
- Nombre de valeurs non-nulles

---

## 4️⃣ Analyse des Variables Catégorielles

### 🔍 **Nutriscore Grade** (`nutriscore_grade`)

- **7 catégories**
- Très déséquilibré (**ratio 62.66**)
- **Majorité inconnue** : `unknown` (156,213 produits)
- **Distribution** : `unknown > e > d > c > a`

### 🔍 **Groupes Alimentaires** (`pnns_groups_1`)

- **11 catégories**
- **Extrêmement déséquilibré** (**ratio 413.13**)
- **Majorité inconnue** : `unknown` (179,300 produits)
- **Principales catégories** :
    1. Sugary snacks
    2. Cereals and potatoes
    3. Fat and sauces
    4. Beverages

📌 Visualisation avec :

- **Graphique en barres** (distribution des catégories)
- **Métriques avancées** :
    - Entropie (diversité des catégories)
    - Ratio de déséquilibre
    - Ratio de valeurs manquantes
    - Ratio de valeurs uniques

---

## 5️⃣ Optimisation de la Mémoire

- **Taille totale** : **34.91 MB**
- **Optimisations réalisées** :  
    ✅ Conversion de `created_t` et `last_modified_t` de `int64` ➡️ `int32`  
    ✅ Optimisation des variables catégorielles

### 📊 **Utilisation de la Mémoire**

- **Camembert** : Répartition mémoire par type de données
- **Barres horizontales** : **Top 10** colonnes consommant le plus de mémoire

---

## 📌 **Résumé Visuel des Résultats**

📌 **Les graphiques permettent maintenant de voir** :  
✅ Répartition des types de variables  
✅ Corrélations entre variables numériques  
✅ Distribution des valeurs numériques (groupées par échelle)