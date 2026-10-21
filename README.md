<div align="center">

# 🩺 Étude du Diabète

### Prédire le diabète à partir de mesures cliniques grâce à un arbre de décision

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Statut](https://img.shields.io/badge/statut-projet%20académique-2ea44f)

<br>

<img src="Graphiques%20diabetes/23.Arbre_de_decision_avec_la_profondeur_choisie.png" alt="Arbre de décision élagué (profondeur 2)" width="70%">

<em>Arbre de décision final (profondeur maximale = 2) : 77,3 % d'accuracy sur la base de test</em>

</div>

---

## 📑 Sommaire

1. [Résumé](#-résumé)
2. [Contexte et objectifs](#-contexte-et-objectifs)
3. [Les données](#-les-données)
4. [Structure du projet](#-structure-du-projet)
5. [Méthodologie détaillée](#-méthodologie-détaillée)
6. [Pourquoi une profondeur de 2 ?](#-pourquoi-une-profondeur-optimale-de-2-)
7. [Résultats du modèle](#-résultats-du-modèle)
8. [Limites et précautions](#-limites-et-précautions)
9. [Installation et utilisation](#-installation-et-utilisation)
10. [Reproductibilité](#-reproductibilité)
11. [Dépannage](#-dépannage)
12. [Pistes d'amélioration](#-pistes-damélioration)
13. [Index des graphiques](#-index-des-graphiques)
14. [Glossaire](#-glossaire)
15. [Références](#-références)
16. [Auteur](#-auteur)

---

## 🧾 Résumé

| | |
|---|---|
| **Problème** | Prédire si une patiente est diabétique à partir de 8 mesures cliniques (glucose, IMC, âge, grossesses, pression artérielle, etc.) |
| **Données** | *Pima Indians Diabetes Database* (Kaggle) : 768 patientes, 8 variables explicatives |
| **Cible** | Binaire : `0` = non diabétique (500, 65,1 %), `1` = diabétique (268, 34,9 %) |
| **Modèle** | Arbre de décision (`DecisionTreeClassifier`), profondeur maximale **2** choisie par validation croisée à 10 plis |
| **Performance (test)** | Accuracy **77,27 %** · AUC **0,77** · pour la classe « diabétique » : précision 0,73 et rappel 0,58 |
| **Point de repère** | Un « modèle » qui répondrait toujours « non diabétique » obtiendrait **64,3 %** : le gain réel est d'environ 13 points |
| **Variable la plus déterminante** | Le **glucose** (≈ 63 % de l'importance), suivi de l'IMC et de l'âge |
| **⚠️ Points de vigilance** | Le rappel de 0,58 signifie que **23 patientes diabétiques sur 55 ne sont pas détectées** ; de nombreuses valeurs `0` sont en réalité des **valeurs manquantes** (voir [Limites](#-limites-et-précautions)) |

---

## 🎯 Contexte et objectifs

Le diabète est une maladie chronique très répandue, dont le dépistage précoce est un enjeu majeur de santé publique. Ce projet explore un jeu de données de mesures cliniques classique en *machine learning* médical, et répond à quatre questions :

1. **Quel est le profil** de la population étudiée ?
2. **Qu'est-ce qui différencie** les patientes diabétiques des autres ?
3. **Ces différences sont-elles statistiquement significatives ?**
4. **Peut-on prédire** le diabète à partir de ces mesures, et avec quelle fiabilité ?

Le script `Arbre_décisionnel_diabetes.py` y répond de bout en bout : exploration, visualisation, tests statistiques, modélisation, optimisation et évaluation.

---

## 📋 Les données

### Source et volumétrie

- **Téléchargement** : [*Pima Indians Diabetes Database* sur Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Origine** : les données proviennent du *National Institute of Diabetes and Digestive and Kidney Diseases* (NIDDK, États-Unis) et ont été mises à disposition via le dépôt UCI (donateur : Vincent Sigillito, Johns Hopkins University). Elles concernent des patientes **d'au moins 21 ans, d'origine Pima, vivant près de Phoenix (Arizona)**.
- **Référence scientifique** : Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C. & Johannes, R.S. (1988), *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*, Proceedings of the Symposium on Computer Applications and Medical Care, pp. 261-265, IEEE Computer Society Press.
- **Définition de la cible** : d'après la documentation d'origine, `Outcome = 1` signifie que la patiente présente des signes de diabète selon les **critères de l'OMS** (glycémie à 2 heures d'au moins 200 mg/dl lors d'un test de tolérance orale au glucose, ou constatée lors de soins courants).
- **Fichier** : `diabetes.csv` · **768 lignes × 9 colonnes** · aucune valeur manquante *au sens informatique* (voir plus bas) · **aucun doublon**.

### Dictionnaire des variables

| Colonne | Description | Plage observée | Moyenne |
|---------|-------------|:--------------:|:-------:|
| `Pregnancies` | Nombre de grossesses | 0 à 17 | 3,85 |
| `Glucose` | Concentration plasmatique de glucose, mesurée lors d'un test de tolérance orale au glucose (2 heures) | 0 à 199 | 120,9 |
| `BloodPressure` | Pression artérielle diastolique (mm Hg) | 0 à 122 | 69,1 |
| `SkinThickness` | Épaisseur du pli cutané tricipital (mm) | 0 à 99 | 20,5 |
| `Insulin` | Insuline sérique à 2 heures (mu U/ml) | 0 à 846 | 79,8 |
| `BMI` | Indice de masse corporelle (kg/m²) | 0 à 67,1 | 32,0 |
| `DiabetesPedigreeFunction` | Score d'hérédité du diabète (« fonction pedigree ») | 0,08 à 2,42 | 0,47 |
| `Age` | Âge en années | 21 à 81 | 33,2 |
| `Outcome` | **Cible** : `1` = diabétique, `0` = non diabétique | 0 ou 1 | 0,35 |

### Répartition de la cible

| Classe | Effectif | Part |
|--------|:--------:|:----:|
| `0` · Non diabétique | 500 | 65,1 % |
| `1` · Diabétique | 268 | 34,9 % |

Le jeu de données est **modérément déséquilibré** : environ deux patientes sur trois ne sont pas diabétiques.

### Qualité des données : points à connaître

- ✅ **Aucun doublon** et **aucune cellule vide**.
- ⚠️ **Les zéros cachent des valeurs manquantes.** Un glucose, une pression artérielle ou un IMC égal à 0 est physiologiquement impossible. Ces zéros représentent en réalité des mesures absentes :

| Variable | Nombre de `0` | Part des 768 lignes |
|----------|:-------------:|:-------------------:|
| `Insulin` | 374 | **48,7 %** |
| `SkinThickness` | 227 | **29,6 %** |
| `BloodPressure` | 35 | 4,6 % |
| `BMI` | 11 | 1,4 % |
| `Glucose` | 5 | 0,7 % |

> Le script **ne corrige pas** ces valeurs (pas d'imputation) : les zéros sont utilisés tels quels par les tests et par le modèle. Un `0` pour `Pregnancies` (111 patientes), en revanche, est une valeur parfaitement légitime.

### Aperçu de la population

Patientes de 21 à 81 ans (médiane 29 ans), avec une forte proportion de jeunes adultes. Les pics à zéro de la pression artérielle, de l'IMC, du pli cutané et de l'insuline sont la signature des valeurs manquantes décrites ci-dessus.

<table>
  <tr>
    <td align="center"><img src="Graphiques%20diabetes/9.Distribution_de_la_variable_cible.png" width="100%"><br><sub><b>Variable cible : diabétique (1) ou non (0)</b></sub></td>
    <td align="center"><img src="Graphiques%20diabetes/2.Distribution_de_la_concentration_de_glucose.png" width="100%"><br><sub><b>Glucose</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="Graphiques%20diabetes/6.Distribution_de_lindice_de_masse_corporelle.png" width="100%"><br><sub><b>Indice de masse corporelle (pic à 0 = valeurs manquantes)</b></sub></td>
    <td align="center"><img src="Graphiques%20diabetes/8.Distribution_de_lage.png" width="100%"><br><sub><b>Âge</b></sub></td>
  </tr>
</table>

<details>
<summary><b>📂 Voir les autres distributions</b> (grossesses, pression artérielle, pli cutané, insuline, fonction pedigree)</summary>

<br>

<table>
  <tr>
    <td align="center"><img src="Graphiques%20diabetes/1.Distribution_du_nombre_de_grossesses.png" width="100%"><br><sub><b>Nombre de grossesses</b></sub></td>
    <td align="center"><img src="Graphiques%20diabetes/3.Distribution_de_la_pression_arterielle_diastolique.png" width="100%"><br><sub><b>Pression artérielle diastolique (pic à 0 = valeurs manquantes)</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="Graphiques%20diabetes/4.Distribution_de_lepaisseur_du_pli_cutane_tricipital.png" width="100%"><br><sub><b>Épaisseur du pli cutané tricipital</b></sub></td>
    <td align="center"><img src="Graphiques%20diabetes/5.Distribution_du_taux_dinsuline_serique_de_2_heures.png" width="100%"><br><sub><b>Insuline sérique à 2 heures</b></sub></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><img src="Graphiques%20diabetes/7.Distribution_de_la_fonction_pedigree_du_diabete.png" width="50%"><br><sub><b>Fonction pedigree du diabète</b></sub></td>
  </tr>
</table>

</details>

---

## 📁 Structure du projet

```text
Pima-Diabetes-Decision-Tree/
├── 📄 Arbre_décisionnel_diabetes.py   # Script complet : analyse + machine learning
├── 📄 diabetes.csv                    # Dataset (768 patientes, 9 colonnes)
├── 📄 README.md                       # Ce fichier
└── 📁 Graphiques diabetes/            # créé automatiquement par le script : 25 graphiques PNG (300 dpi)
```

Le script est organisé en blocs successifs, qui suivent l'ordre de la méthodologie ci-dessous.

---

## 🔬 Méthodologie détaillée

```mermaid
flowchart LR
    A[📥 Chargement<br>du CSV] --> B[🔍 Exploration<br>info, describe]
    B --> C[📊 Distributions<br>9 graphiques]
    C --> D[📦 Boxplots<br>diabétique vs non]
    D --> E[🧪 Tests<br>Kruskal-Wallis]
    E --> F[🔥 Corrélations<br>heatmap]
    F --> G[✂️ Split<br>80 % / 20 %]
    G --> H[🌳 Arbre non élagué<br>référence]
    H --> I[🎯 Validation croisée<br>choix de max_depth]
    I --> J[🌳 Arbre élagué<br>profondeur 2]
    J --> K[📈 Évaluation<br>accuracy, ROC, AUC]
```

### 1. Chargement et exploration

Le CSV est chargé dans un DataFrame pandas. Le script affiche la structure et les types (`info()`), les statistiques descriptives (`describe()`), le nombre de valeurs uniques par colonne (`nunique()`), et les valeurs manquantes (`isnull().sum()`). Le constat sur les zéros (voir [Qualité des données](#qualité-des-données--points-à-connaître)) provient d'une analyse complémentaire du jeu de données : le script, lui, ne les signale pas.

### 2. Préparation des variables

Les 8 variables explicatives sont **déjà numériques** : aucun encodage n'est nécessaire. Aucune normalisation n'est appliquée non plus, car un arbre de décision ne dépend pas de l'échelle des variables (il compare chaque variable à un seuil). Les valeurs `0` suspectes sont conservées (voir [Limites](#-limites-et-précautions)).

| Élément | Contenu |
|---------|---------|
| `X` | `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age` |
| `y` | `Outcome` (0 = non diabétique, 1 = diabétique) |

### 3. Visualisations et sauvegarde automatique

Les histogrammes (`.hist()`) montrent la distribution de chaque variable numérique, et un diagramme en barres celle de la cible. La fonction `save_plot_with_title()` enregistre chaque graphique **sans jamais écraser un fichier précédent** :

1. elle incrémente un compteur global (`plot_counter`) ;
2. elle lit le titre du graphique courant (`plt.gca().get_title()`) et en retire les accents ;
3. elle construit un nom de fichier `compteur.titre` (espaces remplacés par des `_`, guillemets retirés), par exemple `8.Distribution_de_lage.png` : des noms simples, compatibles avec GitHub et tous les systèmes ;
4. elle enregistre le graphique en PNG à **300 dpi** (`bbox_inches="tight"`) dans le dossier **`Graphiques diabetes/`**, créé une seule fois au début du script par `os.makedirs(..., exist_ok=True)` (aucune erreur s'il existe déjà).

### 4. Analyse bivariée : boxplots « non diabétiques vs diabétiques »

Pour chacune des 8 variables, un boxplot compare les deux groupes (`Outcome` = 0 ou 1).

> 📖 **Lire un boxplot** : la ligne centrale est la médiane, la boîte contient 50 % des valeurs (du 1ᵉʳ au 3ᵉ quartile), les moustaches s'étendent jusqu'aux valeurs non aberrantes et les points isolés sont des valeurs atypiques.

<table>
  <tr>
    <td align="center"><img src="Graphiques%20diabetes/11.Boxplot_de_la_variable_Glucose_par_rapport_a_la_variable_cible_Outcome.png" width="100%"><br><sub><b>Glucose</b> · médiane 107 → 140</sub></td>
    <td align="center"><img src="Graphiques%20diabetes/15.Boxplot_de_la_variable_BMI_par_rapport_a_la_variable_cible_Outcome.png" width="100%"><br><sub><b>IMC</b> · médiane 30,1 → 34,3</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="Graphiques%20diabetes/17.Boxplot_de_la_variable_Age_par_rapport_a_la_variable_cible_Outcome.png" width="100%"><br><sub><b>Âge</b> · médiane 27 → 36 ans</sub></td>
    <td align="center"><img src="Graphiques%20diabetes/10.Boxplot_de_la_variable_Pregnancies_par_rapport_a_la_variable_cible_Outcome.png" width="100%"><br><sub><b>Grossesses</b> · médiane 2 → 4</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="Graphiques%20diabetes/16.Boxplot_de_la_variable_DiabetesPedigreeFunction_par_rapport_a_la_variable_cible_Outcome.png" width="100%"><br><sub><b>Fonction pedigree</b> · médiane 0,34 → 0,45</sub></td>
    <td align="center"><img src="Graphiques%20diabetes/12.Boxplot_de_la_variable_BloodPressure_par_rapport_a_la_variable_cible_Outcome.png" width="100%"><br><sub><b>Pression artérielle</b> · médiane 70 → 74</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="Graphiques%20diabetes/13.Boxplot_de_la_variable_SkinThickness_par_rapport_a_la_variable_cible_Outcome.png" width="100%"><br><sub><b>Pli cutané</b> · médiane 21 → 27</sub></td>
    <td align="center"><img src="Graphiques%20diabetes/14.Boxplot_de_la_variable_Insulin_par_rapport_a_la_variable_cible_Outcome.png" width="100%"><br><sub><b>Insuline</b> · médiane 39 → 0 (artefact des zéros)</sub></td>
  </tr>
</table>

| Variable | Médiane (non diabétiques) | Médiane (diabétiques) |
|----------|:-------------------------:|:---------------------:|
| `Glucose` | 107 | **140** |
| `BMI` | 30,05 | **34,25** |
| `Age` | 27 | **36** |
| `Pregnancies` | 2 | **4** |
| `DiabetesPedigreeFunction` | 0,336 | **0,449** |
| `BloodPressure` | 70 | 74 |
| `SkinThickness` | 21 | 27 |
| `Insulin` | 39 | 0 |

**Lecture** : les patientes diabétiques ont en moyenne un **glucose nettement plus élevé**, un **IMC plus haut**, sont **plus âgées** et ont eu **plus de grossesses**.

> 🔎 **Attention à l'insuline.** La médiane des diabétiques (0) est *inférieure* à celle des non diabétiques (39), ce qui n'a aucun sens médical. C'est un **artefact** : environ la moitié des patientes (47 % des non diabétiques, 52 % des diabétiques) ont une insuline à `0`, c'est-à-dire non mesurée, ce qui tire la médiane vers zéro.

### 5. Tests statistiques : Kruskal-Wallis

Les boxplots suggèrent des différences, mais elles pourraient être dues au hasard. Le **test de Kruskal-Wallis** (`scipy.stats.kruskal`) vérifie, pour chaque variable, si sa distribution diffère entre diabétiques et non diabétiques.

- **H₀** : il n'y a pas de différence entre les deux groupes.
- **H₁** : il existe une différence significative.
- **Règle de décision** : si p-value < 0,05, on rejette H₀.

**Pourquoi ce test ?** C'est un test **non paramétrique** : il ne suppose pas que les données suivent une loi normale, ce qui convient ici (distributions très asymétriques, par exemple l'âge, l'insuline ou la fonction pedigree).

| Variable | p-value | Significative (α = 0,05) |
|----------|---------|:------------------------:|
| Glucose | 1,20 × 10⁻³⁹ | ✅ |
| BMI | 9,72 × 10⁻¹⁸ | ✅ |
| Age | 1,14 × 10⁻¹⁷ | ✅ |
| Pregnancies | 3,74 × 10⁻⁸ | ✅ |
| DiabetesPedigreeFunction | 1,20 × 10⁻⁶ | ✅ |
| BloodPressure | 7,55 × 10⁻⁵ | ✅ |
| SkinThickness | 1,30 × 10⁻² | ✅ |
| Insulin | 6,56 × 10⁻² | ❌ |

> ⚠️ **Précautions d'interprétation.** (1) Huit tests sont réalisés : avec une correction de Bonferroni (α = 0,05 / 8 ≈ 0,006), `SkinThickness` (p = 0,013) **cesse d'être significative**, tandis que les six variables les plus fortes restent significatives. (2) `Insulin` est probablement pénalisée par ses nombreux zéros (valeurs non mesurées). (3) Une différence **significative** ne signifie ni **causale** ni **importante en pratique**.

### 6. Matrice de corrélation

La heatmap affiche la **corrélation de Pearson** (de −1 à +1) entre les 8 variables explicatives.

<div align="center">
  <img src="Graphiques%20diabetes/18.Heatmap_de_correlation_des_variables_numeriques.png" alt="Heatmap de corrélation des variables numériques" width="70%">
  <p><em>Corrélation de Pearson entre les 8 variables explicatives</em></p>
</div>

Principaux enseignements :

- 🤰 L'**âge** et le **nombre de grossesses** sont les plus corrélés (r = 0,54), ce qui est logique.
- 📏 Le **pli cutané** est lié à l'**insuline** (r = 0,44) et à l'**IMC** (r = 0,39).
- 🍬 Le **glucose** est modérément lié à l'**insuline** (r = 0,33).
- Aucune paire de variables n'est quasi redondante (aucune corrélation supérieure à 0,6) : l'arbre dispose d'informations variées.

> ℹ️ La corrélation de chaque variable avec la cible (calculée à part, non affichée sur la heatmap) est la plus forte pour le **glucose** (r = 0,47), puis l'**IMC** (0,29), l'**âge** (0,24) et les **grossesses** (0,22). La corrélation de Pearson mesure une relation **linéaire**.

### 7. Préparation de la modélisation

**Séparation apprentissage / test** avec `train_test_split(test_size=0.2, random_state=42)` :

| Ensemble | Effectif | Non diabétiques | Diabétiques | Rôle |
|----------|:--------:|:---------------:|:-----------:|------|
| Apprentissage (train) | 614 (80 %) | 401 | 213 | Construire l'arbre **et** choisir la profondeur |
| Test | 154 (20 %) | 99 | 55 | Évaluer le modèle final sur des données non utilisées pour l'entraînement |

`random_state=42` rend le découpage **reproductible**. Le découpage n'est pas stratifié : les proportions de classes (≈ 65 % / 35 %) sont toutefois bien conservées dans les deux ensembles.

### 8. L'arbre de décision, en bref

Un arbre de décision pose une série de questions simples sur les variables (ex. « le glucose est-il supérieur à 127,5 ? »). Chaque réponse oriente vers une branche, jusqu'à une **feuille** qui donne la prédiction. À chaque nœud, l'algorithme choisit la question qui **sépare le mieux** les deux classes, au sens de l'indice de **Gini** (0 = nœud parfaitement pur, 0,5 = mélange équilibré). Les arbres sont **interprétables**, rapides à entraîner et demandent peu de préparation des données.

**Premier modèle : arbre non élagué (référence).** Sans limite de profondeur, l'arbre grandit jusqu'à ce que chaque feuille soit pure :

| Caractéristique | Valeur |
|-----------------|:------:|
| Profondeur atteinte | 13 |
| Nombre de feuilles | **103** |
| Accuracy train / test | **100 %** / 74,68 % |
| AUC train / test | 1,00 / 0,74 |

> 🔎 **C'est un cas d'école de surapprentissage.** L'arbre classe parfaitement les 614 patientes d'entraînement (100 %) en mémorisant leurs particularités, mais retombe à 74,7 % sur de nouvelles patientes : l'écart de plus de 25 points entre apprentissage et test est le signe qu'il ne généralise pas.

<table>
  <tr>
    <td align="center"><img src="Graphiques%20diabetes/19.Classificateur_par_arbre_de_decision.png" width="100%"><br><sub><b>Arbre non élagué · profondeur 13, 103 feuilles</b></sub></td>
    <td align="center"><img src="Graphiques%20diabetes/21.Receiver_operating_characteristic.png" width="100%"><br><sub><b>Courbes ROC · AUC train 1,00 · test 0,74</b></sub></td>
  </tr>
</table>

---

## 🎯 Pourquoi une profondeur optimale de 2 ?

### Le problème : le compromis biais / variance

La **profondeur maximale** (`max_depth`) fixe le nombre de questions successives que l'arbre peut poser :

- **Trop faible** → l'arbre est trop simple pour capter les motifs des données : c'est le **sous-apprentissage** (*underfitting*).
- **Trop élevée** → l'arbre épouse les particularités de l'échantillon d'entraînement, y compris le bruit : c'est le **surapprentissage** (*overfitting*), comme l'arbre non élagué ci-dessus.

Il faut donc trouver la profondeur qui offre **le meilleur compromis entre performance et simplicité**.

### La méthode : validation croisée à 10 plis

Le script teste **toutes les profondeurs de 1 à 14** et évalue chacune par **validation croisée à 10 plis** (`cross_val_score(..., cv=10)`) :

1. la base d'entraînement (614 patientes) est découpée en 10 blocs ;
2. pour chaque profondeur, l'arbre est entraîné 10 fois, chaque fois sur 9 blocs, puis testé sur le bloc restant ;
3. les 10 scores sont moyennés : c'est le **score de validation croisée** de cette profondeur ;
4. la profondeur retenue est celle qui maximise ce score (`np.argmax`).

> 🔒 **Bonne pratique respectée** : la validation croisée utilise **uniquement la base d'entraînement**. La base de test reste intacte pour l'évaluation finale. Choisir la profondeur sur la base de test reviendrait à « tricher » (fuite d'information).

<div align="center">
  <img src="Graphiques%20diabetes/22.Validation_croisee_pour_larbre_de_decision.png" alt="Validation croisée pour l'arbre de décision" width="70%">
  <p><em>Score moyen de validation croisée (10 plis) selon la profondeur maximale : le maximum est atteint à la profondeur 2 (≈ 0,740), puis le score décline.</em></p>
</div>

### Les résultats, profondeur par profondeur

| Profondeur | Score de validation croisée | Écart-type entre plis | Accuracy train | Accuracy test | Nombre de feuilles |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0,7216 | 0,041 | 0,7345 | 0,7403 | 2 |
| **2** | **0,7395** | 0,045 | 0,7720 | 0,7727 | **4** |
| 3 | 0,7380 | 0,050 | 0,7769 | 0,7597 | 8 |
| 4 | 0,7263 | 0,032 | 0,7980 | 0,6948 | 16 |
| 5 | 0,7279 | 0,047 | 0,8420 | 0,7922 | 27 |
| 6 | 0,7182 | 0,032 | 0,8664 | 0,7468 | 39 |
| 7 | 0,7149 | 0,045 | 0,9104 | 0,7338 | 55 |
| 8 | 0,7166 | 0,037 | 0,9446 | 0,7532 | 73 |
| 9 | 0,7215 | 0,034 | 0,9691 | 0,7662 | 85 |
| 10 | 0,7182 | 0,036 | 0,9837 | 0,7532 | 94 |
| 11 | 0,7214 | 0,039 | 0,9967 | 0,7468 | 101 |
| 12 | 0,7167 | 0,035 | 0,9984 | 0,7403 | 102 |
| 13 | 0,7183 | 0,031 | 1,0000 | 0,7468 | 103 |
| 14 | 0,7101 | 0,040 | 1,0000 | 0,7468 | 103 |

> Les colonnes « Accuracy test » sont données **à titre d'information uniquement** : le choix de la profondeur ne repose que sur la validation croisée.

### Comment lire cette courbe

1. **Profondeur 1 : sous-apprentissage léger.** Une seule question ne suffit pas (score 0,722).
2. **Profondeur 2 : le meilleur score** (**0,7395**). Avec deux niveaux de questions, l'arbre capte l'essentiel du signal.
3. **Profondeur 3 : score quasi identique** (0,7380), puis un **déclin** : au-delà, le score de validation croisée oscille autour de 0,71 à 0,73.
4. **Surapprentissage progressif.** L'accuracy d'entraînement grimpe de 77 % à **100 %**, mais le score de validation croisée **n'augmente pas** : les niveaux supplémentaires mémorisent du bruit sans améliorer la généralisation.

### Pourquoi 2 et pas plus ?

Ici, contrairement à un cas d'égalité, **le maximum est unique** : la profondeur 2 obtient le meilleur score de validation croisée, devant la profondeur 3 (écart de 0,0015, négligeable). Le principe de parcimonie (rasoir d'Occam) confirme le choix : à performance équivalente, on préfère le modèle le plus simple, ici un arbre de **4 feuilles seulement** (contre 8 à la profondeur 3 et 103 sans limite).

### ⚖️ Nuance importante : un optimum « raisonnable », pas « unique »

- L'écart-type entre plis est d'environ **0,045**, soit une erreur standard d'environ **0,014**. Les profondeurs **2 à 5** sont toutes à moins d'une erreur standard du meilleur score : elles sont **statistiquement difficiles à distinguer**. La profondeur 2 reste le choix le plus simple dans cette zone (la règle de l'erreur standard conduit au même résultat).
- Sur la base de test, la profondeur 5 donne 79,2 % contre 77,3 % pour la profondeur 2, soit **3 patientes de plus** bien classées sur 154. Cet écart est dans la marge d'incertitude d'un seul découpage : avec 154 patientes de test, l'accuracy est connue à environ **± 7 points** (intervalle à 95 %). Retenir la profondeur d'après le test serait de plus une fuite d'information.
- Le modèle de profondeur 2 est enfin **entièrement lisible** (voir ci-dessous), ce qui est un atout pour un usage médical.

> ✅ **En résumé** : la profondeur 2 est retenue parce qu'elle atteint le **meilleur score de validation croisée avec le plus petit arbre**. Elle évite le surapprentissage qui fait chuter la performance de l'arbre non élagué.

---

## 📈 Résultats du modèle

### Les règles de l'arbre final

<div align="center">
  <img src="Graphiques%20diabetes/23.Arbre_de_decision_avec_la_profondeur_choisie.png" alt="Arbre de décision élagué (profondeur 2)" width="75%">
  <p><em>Arbre de décision final (profondeur 2). Chaque case indique la question, l'indice de Gini, le nombre de patientes et leur répartition [non diabétiques, diabétiques].</em></p>
</div>

| Chemin dans l'arbre | Patientes d'entraînement | Prédiction |
|---------------------|:------------------------:|:----------:|
| `Glucose ≤ 127,5` (puis `Age ≤ 28,5` ou non) | 221 + 169 = 390 | **Non diabétique** |
| `Glucose > 127,5` et `BMI ≤ 29,95` | 61 (19 diabétiques) | **Non diabétique** |
| `Glucose > 127,5` et `BMI > 29,95` | 163 (118 diabétiques, soit 72 %) | **Diabétique** |

> 💡 **En pratique, la décision se résume à une règle** : *glucose supérieur à 127,5 **et** IMC supérieur à 29,95 → diabétique*. La question sur l'âge (`Age ≤ 28,5`) améliore la pureté des nœuds mais **ne change aucune prédiction** : ses deux feuilles aboutissent à « non diabétique ».

Cette règle est cohérente avec les données : dans l'ensemble du jeu de données, **61,5 %** des patientes avec un glucose > 127,5 sont diabétiques (contre 19,4 % pour les autres), et **46,2 %** de celles dont l'IMC dépasse 30 (contre 17,5 %).

### Comparaison avant / après élagage

| | Arbre non élagué | **Arbre élagué (profondeur 2)** |
|---|:---:|:---:|
| Profondeur | 13 | **2** |
| Feuilles | 103 | **4** |
| Accuracy train | 100 % | 77,20 % |
| Accuracy test | 74,68 % | **77,27 %** |
| AUC train | 1,00 | 0,80 |
| AUC test | 0,74 | **0,77** |
| Précision (diabétiques) | 0,62 | **0,73** |
| Rappel (diabétiques) | **0,73** | 0,58 |

À retenir, honnêtement :
- ✅ L'élagage **améliore la généralisation** (accuracy de test +2,6 points, AUC +0,03) et rapproche enfin les performances d'entraînement (77,2 %) et de test (77,3 %) : plus de surapprentissage.
- ⚠️ En contrepartie, le modèle élagué **détecte moins de patientes diabétiques** (rappel 0,58 contre 0,73) : il est plus prudent et fait moins de fausses alertes (précision 0,73 contre 0,62). Pour un dépistage, c'est un point faible (voir [Limites](#-limites-et-précautions)).

### Performances du modèle final (profondeur 2)

| Métrique | Train | Test |
|----------|:-----:|:----:|
| **Accuracy** | 77,20 % | **77,27 %** |
| **AUC ROC** | 0,80 | **0,77** |

<table>
  <tr>
    <td align="center"><img src="Graphiques%20diabetes/21.Receiver_operating_characteristic.png" width="100%"><br><sub><b>Courbes ROC · arbre non élagué (AUC train 1,00 · test 0,74)</b></sub></td>
    <td align="center"><img src="Graphiques%20diabetes/25.Receiver_operating_characteristic.png" width="100%"><br><sub><b>Courbes ROC · arbre élagué, profondeur 2 (AUC train 0,80 · test 0,77)</b></sub></td>
  </tr>
</table>

**Matrice de confusion (test, 154 patientes)**

|  | Prédit : non diabétique | Prédit : diabétique |
|--|:-----------------------:|:-------------------:|
| **Réel : non diabétique** | 87 (vrais négatifs) | 12 (faux positifs) |
| **Réel : diabétique** | 23 (faux négatifs) | 32 (vrais positifs) |

**Rapport de classification (test)**

| Classe | Précision | Rappel | F1-score | Support |
|--------|:---------:|:------:|:--------:|:-------:|
| 0 · Non diabétique | 0,79 | 0,88 | 0,83 | 99 |
| 1 · Diabétique | 0,73 | 0,58 | 0,65 | 55 |

### 📖 Comment lire ces métriques

| Métrique | Définition | Dans ce projet |
|----------|-----------|----------------|
| **Accuracy** | Part de prédictions correctes | 119 prédictions justes sur 154 (77,3 %) |
| **Précision** (classe « diabétique ») | Quand le modèle annonce un diabète, a-t-il raison ? | 0,73 : 32 alertes sur 44 sont justes |
| **Rappel** (classe « diabétique ») | Parmi les vraies diabétiques, combien sont détectées ? | 0,58 : 32 détectées sur 55 |
| **F1-score** | Moyenne harmonique de la précision et du rappel | 0,65 |
| **AUC ROC** | Capacité à classer une patiente diabétique au-dessus d'une non diabétique (0,5 = hasard, 1 = parfait) | 0,77 |

**Repère indispensable** : une réponse systématique « non diabétique » donnerait **64,3 %** d'accuracy sur la base de test (99 / 154). Le modèle gagne donc environ **13 points**, un apport réel mais modeste. **Lecture clinique** : le modèle commet **23 faux négatifs** (patientes diabétiques jugées saines) contre **12 faux positifs**. Pour un outil de dépistage, les faux négatifs sont l'erreur la plus coûteuse : le rappel (0,58) est donc la métrique à surveiller en priorité.

### 🏆 Importance des variables

<table>
  <tr>
    <td align="center"><img src="Graphiques%20diabetes/20.Importance_des_variables.png" width="100%"><br><sub><b>Importance des variables · arbre non élagué</b></sub></td>
    <td align="center"><img src="Graphiques%20diabetes/24.Importance_des_variables.png" width="100%"><br><sub><b>Importance des variables · arbre élagué (profondeur 2)</b></sub></td>
  </tr>
</table>

| Variable | Importance (arbre élagué) | Importance (arbre non élagué) |
|----------|:-------------------------:|:-----------------------------:|
| **Glucose** | **0,628** | 0,314 |
| BMI | 0,192 | 0,171 |
| Age | 0,180 | 0,099 |
| BloodPressure | 0,000 | 0,148 |
| DiabetesPedigreeFunction | 0,000 | 0,083 |
| Insulin | 0,000 | 0,066 |
| SkinThickness | 0,000 | 0,061 |
| Pregnancies | 0,000 | 0,057 |

Le **glucose** domine dans les deux modèles. L'arbre non élagué répartit l'importance sur toutes les variables (signe qu'il s'appuie sur du bruit), alors que l'arbre élagué se concentre sur trois variables.

> ⚠️ **Comment interpréter l'importance.** (1) L'importance de **Gini** mesure ce que l'arbre *utilise* (la réduction d'impureté), pas ce qui *cause* le diabète. Ainsi, `Age` a une importance de 0,18 alors qu'il ne change aucune prédiction (voir plus haut). (2) Les variables **corrélées entre elles se « partagent » l'information** : une importance nulle ne signifie pas « sans lien » (la pression artérielle, l'hérédité et les grossesses sont significativement associées au diabète d'après le test de Kruskal-Wallis).

---

## 🚧 Limites et précautions

Cette section est essentielle pour interpréter correctement les résultats.

### 1. 0️⃣ Valeurs manquantes déguisées en zéros

Près de **la moitié** des valeurs d'insuline (48,7 %) et près d'un **tiers** de celles du pli cutané (29,6 %) sont à `0`, tout comme quelques dizaines de pressions artérielles et d'IMC. Ces zéros faussent les statistiques (médiane d'insuline, test de Kruskal-Wallis) et peuvent perturber les seuils appris par l'arbre. Le script ne les traite pas : un prétraitement (imputation, valeurs manquantes explicites) est la première amélioration à apporter.

### 2. 🩸 Un taux de détection insuffisant pour du dépistage

Avec un **rappel de 0,58**, le modèle laisse passer plus de 4 patientes diabétiques sur 10 (23 sur 55 dans la base de test). En l'état, il ne peut pas servir d'outil de dépistage. Le **gain par rapport à la réponse « toujours non »** (+13 points d'accuracy) reste modeste.

### 3. 🔁 Le glucose est proche du critère diagnostique

La variable `Glucose` correspond (d'après la documentation du jeu de données) à une mesure de **glucose à 2 heures lors d'un test de tolérance orale au glucose**, un examen qui sert précisément à diagnostiquer le diabète. Or la cible est elle-même définie, selon la documentation d'origine, par les critères de l'OMS fondés sur la glycémie à 2 heures (au moins 200 mg/dl). Le fait que `Glucose` domine le modèle est donc attendu : l'arbre ne « découvre » pas un facteur de risque indépendant, il reprend en grande partie un critère de diagnostic.

### 4. 📉 Petit échantillon et découpage unique

- 768 patientes seulement, dont **154 en test** : chaque erreur pèse pour 0,65 point d'accuracy et l'incertitude est d'environ ± 7 points.
- Un seul découpage train/test est utilisé et il n'est pas **stratifié** : les résultats dépendent du tirage (`random_state=42`). La performance de l'arbre de profondeur 5 sur ce test (79,2 %) montre à quel point un seul découpage peut faire varier les chiffres.

### 5. 🌍 Population très spécifique

Les patientes sont des **femmes d'au moins 21 ans, d'origine Pima**. Les relations observées et les seuils appris (glucose 127,5 ; IMC 29,95) ne sont **pas transposables** à d'autres populations (hommes, autres origines, enfants).

### 6. 🎯 Périmètre du modèle

- **Corrélation n'est pas causalité** : le modèle ne dit pas que l'IMC ou l'âge *provoquent* le diabète.
- Le modèle repose sur **deux seuils** : il ne tient pas compte de l'ensemble des facteurs cliniques.

### 7. 🩺 Avertissement médical

> Ce projet est un **travail d'analyse de données à but éducatif**. Il **ne constitue pas un outil de diagnostic** et ne remplace en aucun cas l'avis d'un professionnel de santé.

---

## 🚀 Installation et utilisation

### Prérequis

- **Python 3.10 ou supérieur** (testé avec Python 3.12)
- `pip`

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/julesfrichetpro-tech/Pima-Diabetes-Decision-Tree.git
cd Pima-Diabetes-Decision-Tree

# 2. (Recommandé) Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install pandas numpy scikit-learn seaborn matplotlib scipy
```

### Bibliothèques utilisées

| Bibliothèque | Rôle dans le projet |
|--------------|---------------------|
| `pandas` | Chargement et manipulation des données |
| `numpy` | Calcul numérique (`argmax`, `argsort`, `mean`) |
| `scikit-learn` | Arbre de décision, split train/test, validation croisée, métriques (accuracy, matrice de confusion, ROC/AUC) |
| `seaborn` | Boxplots et heatmap |
| `matplotlib` | Histogrammes, tracé des arbres, sauvegarde des figures |
| `scipy` | Test de Kruskal-Wallis |

### Lancement

Exécutez le script **depuis le dossier qui contient `diabetes.csv`** (le fichier est lu par un chemin relatif). Le dossier `Graphiques diabetes/` est créé au même endroit :

```bash
python Arbre_décisionnel_diabetes.py
```

### Ce que fait le script à l'exécution

1. Affiche les informations générales, les statistiques descriptives, le nombre de valeurs uniques et les valeurs manquantes dans la console.
2. Affiche chaque graphique dans une fenêtre (`plt.show()`) : **fermez la fenêtre pour passer à la suite**.
3. Crée automatiquement le dossier `Graphiques diabetes/` et y sauvegarde chaque graphique (**25 fichiers**, préfixés par un numéro d'ordre).
4. Affiche le tableau des p-values, les performances de l'arbre non élagué, la **profondeur optimale** trouvée et les performances de l'arbre élagué.

> ℹ️ **À savoir.** Lancez le script sur un poste avec **affichage graphique** (PyCharm, VS Code, terminal local). Sur un serveur sans écran, `plt.show()` ne referme pas les figures et les graphiques se superposeraient.

---

## 🔁 Reproductibilité

- **Graine aléatoire** : `random_state=42` pour le découpage train/test et pour tous les arbres. Les résultats sont identiques d'une exécution à l'autre.
- **Environnement testé** :

| Composant | Version |
|-----------|---------|
| Python | 3.12.3 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| seaborn | 0.13.2 |
| matplotlib | 3.10.8 |

> Les 25 graphiques du dossier `Graphiques diabetes/` ont été générés avec ces versions.
>
> Des versions différentes des bibliothèques peuvent produire de légères variations (valeurs de l'AUC, structure exacte de l'arbre en cas d'égalités).

---

## 🔧 Dépannage

| Problème | Cause probable | Solution |
|----------|----------------|----------|
| `FileNotFoundError` sur `diabetes.csv` | Script lancé depuis un autre dossier que celui du CSV | Se placer dans le dossier qui contient `diabetes.csv` avant de lancer le script |
| Le script « se fige » | Une fenêtre de graphique est ouverte | Fermer la fenêtre du graphique pour passer au suivant |
| Plusieurs graphiques mélangés sur la même image | Exécution sans écran (serveur) : `plt.show()` ne referme pas les figures | Exécuter le script sur un poste avec affichage graphique |
| `ModuleNotFoundError` | Bibliothèque non installée | Relancer la commande `pip install ...` |
| Le dossier `Graphiques diabetes/` est introuvable | Il est créé dans le dossier depuis lequel le script est lancé | Chercher le dossier à côté de `diabetes.csv` |

---

## 🔭 Pistes d'amélioration

Classées par priorité :

1. **Traiter les valeurs manquantes** : remplacer les `0` impossibles par des valeurs manquantes (`NaN`), puis les imputer (médiane, k plus proches voisins) ou utiliser un modèle qui les gère nativement (`HistGradientBoostingClassifier`).
2. **Améliorer le rappel** : pondérer les classes (`class_weight="balanced"`) ou abaisser le seuil de décision, afin de détecter davantage de patientes diabétiques, quitte à accepter plus de fausses alertes.
3. **Stratifier le split** (`stratify=y`) et utiliser une **validation croisée répétée** sur l'ensemble du jeu de données plutôt qu'un seul découpage de 154 patientes.
4. **Comparer d'autres modèles** : forêt aléatoire, gradient boosting, régression logistique.
5. **Optimiser plusieurs hyperparamètres** avec `GridSearchCV` (`criterion`, `min_samples_leaf`, `ccp_alpha`) et tester la règle de l'erreur standard pour le choix de la profondeur.
6. **Étudier l'effet de la variable `Glucose`** en la retirant, pour mesurer ce que les autres variables permettent de prédire seules.
7. **Structurer le code** en modules (`data.py`, `plots.py`, `model.py`) et ajouter un fichier `requirements.txt`.

---

## 📸 Index des graphiques

Le dossier [`Graphiques diabetes/`](Graphiques%20diabetes) contient les **25 graphiques** produits par le script. Le préfixe numérique correspond à l'ordre de génération.

| N° | Fichier | Contenu |
|:--:|---------|---------|
| 1 | [`1.Distribution_du_nombre_de_grossesses.png`](Graphiques%20diabetes/1.Distribution_du_nombre_de_grossesses.png) | Nombre de grossesses |
| 2 | [`2.Distribution_de_la_concentration_de_glucose.png`](Graphiques%20diabetes/2.Distribution_de_la_concentration_de_glucose.png) | Glucose |
| 3 | [`3.Distribution_de_la_pression_arterielle_diastolique.png`](Graphiques%20diabetes/3.Distribution_de_la_pression_arterielle_diastolique.png) | Pression artérielle diastolique |
| 4 | [`4.Distribution_de_lepaisseur_du_pli_cutane_tricipital.png`](Graphiques%20diabetes/4.Distribution_de_lepaisseur_du_pli_cutane_tricipital.png) | Pli cutané tricipital |
| 5 | [`5.Distribution_du_taux_dinsuline_serique_de_2_heures.png`](Graphiques%20diabetes/5.Distribution_du_taux_dinsuline_serique_de_2_heures.png) | Insuline sérique à 2 h |
| 6 | [`6.Distribution_de_lindice_de_masse_corporelle.png`](Graphiques%20diabetes/6.Distribution_de_lindice_de_masse_corporelle.png) | Indice de masse corporelle |
| 7 | [`7.Distribution_de_la_fonction_pedigree_du_diabete.png`](Graphiques%20diabetes/7.Distribution_de_la_fonction_pedigree_du_diabete.png) | Fonction pedigree du diabète |
| 8 | [`8.Distribution_de_lage.png`](Graphiques%20diabetes/8.Distribution_de_lage.png) | Âge |
| 9 | [`9.Distribution_de_la_variable_cible.png`](Graphiques%20diabetes/9.Distribution_de_la_variable_cible.png) | Variable cible (diabétique ou non) |
| 10 | [`10.Boxplot_de_la_variable_Pregnancies_par_rapport_a_la_variable_cible_Outcome.png`](Graphiques%20diabetes/10.Boxplot_de_la_variable_Pregnancies_par_rapport_a_la_variable_cible_Outcome.png) | Boxplot : grossesses |
| 11 | [`11.Boxplot_de_la_variable_Glucose_par_rapport_a_la_variable_cible_Outcome.png`](Graphiques%20diabetes/11.Boxplot_de_la_variable_Glucose_par_rapport_a_la_variable_cible_Outcome.png) | Boxplot : glucose |
| 12 | [`12.Boxplot_de_la_variable_BloodPressure_par_rapport_a_la_variable_cible_Outcome.png`](Graphiques%20diabetes/12.Boxplot_de_la_variable_BloodPressure_par_rapport_a_la_variable_cible_Outcome.png) | Boxplot : pression artérielle |
| 13 | [`13.Boxplot_de_la_variable_SkinThickness_par_rapport_a_la_variable_cible_Outcome.png`](Graphiques%20diabetes/13.Boxplot_de_la_variable_SkinThickness_par_rapport_a_la_variable_cible_Outcome.png) | Boxplot : pli cutané |
| 14 | [`14.Boxplot_de_la_variable_Insulin_par_rapport_a_la_variable_cible_Outcome.png`](Graphiques%20diabetes/14.Boxplot_de_la_variable_Insulin_par_rapport_a_la_variable_cible_Outcome.png) | Boxplot : insuline |
| 15 | [`15.Boxplot_de_la_variable_BMI_par_rapport_a_la_variable_cible_Outcome.png`](Graphiques%20diabetes/15.Boxplot_de_la_variable_BMI_par_rapport_a_la_variable_cible_Outcome.png) | Boxplot : IMC |
| 16 | [`16.Boxplot_de_la_variable_DiabetesPedigreeFunction_par_rapport_a_la_variable_cible_Outcome.png`](Graphiques%20diabetes/16.Boxplot_de_la_variable_DiabetesPedigreeFunction_par_rapport_a_la_variable_cible_Outcome.png) | Boxplot : fonction pedigree |
| 17 | [`17.Boxplot_de_la_variable_Age_par_rapport_a_la_variable_cible_Outcome.png`](Graphiques%20diabetes/17.Boxplot_de_la_variable_Age_par_rapport_a_la_variable_cible_Outcome.png) | Boxplot : âge |
| 18 | [`18.Heatmap_de_correlation_des_variables_numeriques.png`](Graphiques%20diabetes/18.Heatmap_de_correlation_des_variables_numeriques.png) | Matrice de corrélation (heatmap) |
| 19 | [`19.Classificateur_par_arbre_de_decision.png`](Graphiques%20diabetes/19.Classificateur_par_arbre_de_decision.png) | Arbre de décision non élagué |
| 20 | [`20.Importance_des_variables.png`](Graphiques%20diabetes/20.Importance_des_variables.png) | Importance des variables (arbre non élagué) |
| 21 | [`21.Receiver_operating_characteristic.png`](Graphiques%20diabetes/21.Receiver_operating_characteristic.png) | Courbes ROC (arbre non élagué) |
| 22 | [`22.Validation_croisee_pour_larbre_de_decision.png`](Graphiques%20diabetes/22.Validation_croisee_pour_larbre_de_decision.png) | Validation croisée : score selon la profondeur |
| 23 | [`23.Arbre_de_decision_avec_la_profondeur_choisie.png`](Graphiques%20diabetes/23.Arbre_de_decision_avec_la_profondeur_choisie.png) | Arbre de décision élagué (profondeur 2) |
| 24 | [`24.Importance_des_variables.png`](Graphiques%20diabetes/24.Importance_des_variables.png) | Importance des variables (arbre élagué) |
| 25 | [`25.Receiver_operating_characteristic.png`](Graphiques%20diabetes/25.Receiver_operating_characteristic.png) | Courbes ROC (arbre élagué) |

---

## 📚 Glossaire

| Terme | Définition |
|-------|-----------|
| **Arbre de décision** | Modèle qui prédit en posant une suite de questions simples sur les variables |
| **Feuille** | Extrémité d'un arbre, qui donne la prédiction finale |
| **Indice de Gini** | Mesure d'impureté d'un nœud : 0 = une seule classe, 0,5 = mélange équilibré (2 classes) |
| **Élagage** | Limitation de la taille de l'arbre (ici via `max_depth`) pour éviter le surapprentissage |
| **Surapprentissage** | Modèle trop adapté aux données d'entraînement, qui généralise mal |
| **Sous-apprentissage** | Modèle trop simple pour capter les motifs des données |
| **Validation croisée** | Évaluation répétée sur plusieurs découpages de la base d'entraînement |
| **Accuracy** | Proportion de prédictions correctes |
| **Précision / Rappel** | Fiabilité des alertes / part des vrais cas détectés |
| **Faux négatif** | Patiente diabétique que le modèle juge non diabétique |
| **Courbe ROC / AUC** | Compromis entre vrais et faux positifs selon le seuil ; l'AUC résume la courbe (0,5 = hasard, 1 = parfait) |
| **Kruskal-Wallis** | Test non paramétrique qui compare la distribution d'une variable entre groupes |
| **p-value** | Probabilité d'observer un écart au moins aussi grand si H₀ était vraie |
| **IMC / BMI** | Indice de masse corporelle (poids / taille²) |
| **Valeur manquante déguisée** | Valeur absente codée par un chiffre (ici `0`) au lieu d'une cellule vide |
| **Fuite d'information** | Utilisation, à l'entraînement, d'informations qui ne seraient pas disponibles en situation réelle |

---

## 📖 Références

- Jeu de données : [*Pima Indians Diabetes Database*, Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C. & Johannes, R.S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. Proceedings of the Symposium on Computer Applications and Medical Care, pp. 261-265. IEEE Computer Society Press.
- [scikit-learn : arbres de décision](https://scikit-learn.org/stable/modules/tree.html)
- [SciPy : `scipy.stats.kruskal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)

---

## 👤 Auteur

**Jules Frichet** · [@julesfrichetpro-tech](https://github.com/julesfrichetpro-tech)

---

<div align="center">

⭐ Si ce projet vous a été utile, n'hésitez pas à lui laisser une étoile !

</div>
