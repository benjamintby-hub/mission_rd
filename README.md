# Prédire la kinésiophobie à partir de l'accélérométrie

Projet de R&D (IMT, 2ᵉ année, filière IA), réalisé par **Benjamin Tamby** et **Yasser Salhi**.

Chez les patients lombalgiques chroniques, la peur du mouvement (kinésiophobie) se mesure
avec un questionnaire, le score **TAMPA** (17 à 68). L'objectif est de savoir si l'on peut
**prédire ce score à partir du comportement moteur mesuré en vie réelle**, grâce à un
accéléromètre porté pendant une semaine.

- **Données** : 66 patients, accéléromètre tri-axial à 100 Hz (~60 millions de points par patient) et 5 variables cliniques (âge, sexe, poids, taille, IMC)
- **Méthode** : détection de l'inactivité, puis 118 descripteurs par patient (statistiques, variabilité glissante, DFA, entropie), puis régression (KNN, SVM, arbre, Random Forest) évaluée en Leave-One-Out
- **Résultat** : le signal accélérométrique apporte une information réelle mais faible. Un Random Forest sur tous les descripteurs ne bat la baseline que de justesse (MAE 6,39 contre 6,41). Les variables cliniques, surtout l'IMC, sont les plus prédictives.

Le détail du projet (choix techniques, résultats, limites et data leakage) est dans
**[docs/PROJET.md](docs/PROJET.md)**.

## Structure

```
.
├── 01_extraction_descripteurs.Rmd   # Étape 1 : .gt3x bruts -> data/matrice_descripteurs.csv
├── 02_modelisation.R                # Étape 2 : exploration, ACP, clustering, ML, interprétation
├── data/                            # Données patients (NON versionnées, voir data/README.md)
├── outputs/                         # Figures PDF et CSV générés (non versionnés)
└── docs/PROJET.md                   # Documentation détaillée du projet
```

## Exécution

Prérequis : R ≥ 4.2 et les packages suivants.

```r
install.packages(c("read.gt3x", "tibble", "RcppRoll", "moments", "nonlinearTseries",
                   "entropy", "rmarkdown", "dplyr", "e1071", "FNN", "rpart",
                   "rpart.plot", "randomForest", "ggplot2", "FactoMineR", "factoextra"))
```

Depuis la racine du projet :

```r
rmarkdown::render("01_extraction_descripteurs.Rmd")   # long : plusieurs minutes par patient
source("02_modelisation.R")                            # ~1 h (grilles d'hyperparamètres en LOO)
```

Les données ne sont pas fournies : ce sont des données de santé issues d'une étude clinique.
Le format attendu est décrit dans [data/README.md](data/README.md).
