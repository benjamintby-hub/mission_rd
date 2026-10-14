# Documentation du projet

> Prédiction du score de kinésiophobie TAMPA chez des patients lombalgiques chroniques à
> partir de données accélérométriques en vie réelle.
> Mission R&D, IMT, 2ᵉ année, filière IA. Benjamin Tamby et Yasser Salhi, 2026.

## Sommaire

1. [Le projet en 30 secondes](#1-le-projet-en-30-secondes)
2. [Contexte et question](#2-contexte-et-question)
3. [Données](#3-données)
4. [Structure du dépôt](#4-structure-du-dépôt)
5. [Étape 1 : du signal brut aux descripteurs](#5-étape-1--du-signal-brut-aux-descripteurs)
6. [Étape 2 : exploration et modélisation](#6-étape-2--exploration-et-modélisation)
7. [Résultats](#7-résultats)
8. [Limites, dont le data leakage](#8-limites-dont-le-data-leakage)
9. [Ce que je referais autrement](#9-ce-que-je-referais-autrement)
10. [Questions probables et réponses](#10-questions-probables-et-réponses)
11. [Glossaire](#11-glossaire)

---

## 1. Le projet en 30 secondes

On a cherché à prédire un score psychologique, la peur du mouvement (score TAMPA, de 17 à
68), à partir de la façon dont les patients bougent au quotidien. Chaque patient a porté un
accéléromètre pendant une semaine. On a transformé environ 60 millions de mesures par
patient en 118 descripteurs (intensité, variabilité, complexité, entropie), puis entraîné
plusieurs modèles de régression évalués en Leave-One-Out, ce qui convient à un petit
effectif de 66 patients.

**Conclusion honnête** : l'accélérométrie contient une information réelle mais faible. Sans
sélection de variables, le meilleur modèle (Random Forest) fait à peine mieux qu'un modèle
qui prédit toujours la moyenne (MAE 6,39 contre 6,41). Les variables cliniques, l'IMC en
tête, sont plus prédictives. Les meilleurs chiffres du rapport (MAE 5,69 avec 40 variables
sélectionnées) sont optimistes à cause d'une fuite de données dans la sélection de
variables, que j'ai identifiée et documentée (voir §8).

## 2. Contexte et question

- **Lombalgie chronique** : douleur du bas du dos qui dure plus de 3 mois. Elle s'accompagne
  souvent de **kinésiophobie**, une peur de bouger par crainte de se faire mal. Le patient
  évite alors le mouvement, se déconditionne, et la douleur s'entretient.
- **Score TAMPA** (Tampa Scale of Kinesiophobia) : questionnaire de 17 items notés de 1 à 4,
  soit un total de 17 à 68. Le seuil clinique se situe autour de 37–40.
- **Problème** : c'est un auto-questionnaire, donc subjectif (désirabilité sociale,
  fluctuations d'humeur) et ponctuel.
- **Question de recherche** : le comportement moteur mesuré passivement en vie réelle
  permet-il d'estimer ce score ? Si oui, on disposerait d'un suivi continu et objectif.
- **Type de tâche ML** : **régression supervisée** (cible continue), en petit effectif et
  grande dimension (n = 66, p ≈ 120), donc très exposée au sur-apprentissage.

## 3. Données

| Source | Contenu | Volume |
|---|---|---|
| Accéléromètre ActiGraph (`.gt3x`) | Accélération sur X, Y, Z en **g**, à 100 Hz, pendant ~7 jours | ~60 M lignes × 3 axes par patient |
| Base clinique (REDCap) | âge, sexe, poids, taille, IMC, EVA douleur (nuit, dos, jambe), Oswestry, 17 items TAMPA et score total | 92 lignes |

- La jointure (*inner join*) se fait sur l'identifiant `sid`. Sur 91 patients avec un score
  TAMPA valide, **66** ont aussi un enregistrement accélérométrique.
- TAMPA dans la cohorte : min 24, max 62, moyenne 44,8, médiane 45,5, écart-type 8,0. La
  majorité des patients est au-dessus du seuil clinique.
- **Variables exclues des modèles** : les EVA douleur et le score Oswestry. Ils sont
  fortement corrélés au TAMPA et mesurés par le même type de questionnaire au même moment.
  Les utiliser reviendrait à prédire un questionnaire avec un autre questionnaire, ce qui
  ne répond pas à la question « le mouvement prédit-il la peur ? ». Les 17 items TAMPA sont
  évidemment exclus aussi : ils constituent la cible.
- **Confidentialité** : ce sont des données de santé (étude clinique, CHU de Nîmes). Elles
  ne sont pas versionnées dans le dépôt.

## 4. Structure du dépôt

```
.
├── README.md                        # Présentation rapide et instructions
├── 01_extraction_descripteurs.Rmd   # Étape 1 : .gt3x -> data/matrice_descripteurs.csv
├── 02_modelisation.R                # Étape 2 : analyse descriptive + ML + interprétation
├── data/README.md                   # Format attendu des données (non fournies)
├── outputs/                         # PDF et CSV générés (non versionnés)
└── docs/PROJET.md                   # Ce document
```

Le pipeline est linéaire : `data/gt3x/*.gt3x` → **01** → `data/matrice_descripteurs.csv`
(66 × 119) → **02** (avec `data/matrice_clinique.csv`) → `outputs/*.pdf`.

Les figures de `02_modelisation.R` sont numérotées dans l'ordre du raisonnement :

| N° | Figure | À quoi elle sert |
|---|---|---|
| 1–2 | Distributions des variables cliniques et de TAMPA | Connaître la cible et sa dispersion |
| 3–5 | ACP (colorée par TAMPA, par non-port, cercle des corrélations) | Voir s'il existe une structure linéaire |
| 6–8 | Clustering (silhouette, K-means, Ward) | Chercher des profils de patients |
| 9 | Comparaison des 5 modèles × 3 configurations | Choisir le modèle de référence |
| 10–13 | Calibration KNN, SVM, arbre, RF | Hyperparamètres |
| 14 / 14b | Importance %IncMSE (combiné / accéléro seul) | Variables les plus utiles |
| 9b / 13b | RF top 40 | Effet de la sélection de variables |
| 17 | MAE selon le nombre de composantes ACP | Réduction non supervisée |
| 18–19 | Erreurs sur l'ACP, prédit vs réel | Où le modèle se trompe |
| 20–24 | Mal prédits : clusters, Wilcoxon, arbre | Qui sont les patients mal prédits |
| 25 | Arbre clinique interprétable | Règle simple pour un clinicien |

## 5. Étape 1 : du signal brut aux descripteurs

Fichier : `01_extraction_descripteurs.Rmd`. Pour chaque patient :

### 5.1 Norme du vecteur accélération

`norme = sqrt(X² + Y² + Z²)`. Elle est indépendante de l'orientation du capteur et vaut
environ 1 g au repos (gravité).

### 5.2 Détection de l'inactivité (sommeil + non-port)

1. **Décimation 1/100** : on garde un point par seconde (100 Hz → 1 Hz) pour accélérer le
   calcul. Cela suffit pour repérer des périodes de plusieurs minutes.
2. **Écart-type glissant de la norme** sur 5 minutes. S'il est inférieur à **0,01 g**, le
   capteur ne bouge pas : la période est marquée inactive (sommeil ou capteur posé).
3. **Lissage par RLE** (*run-length encoding*) :
   - un éveil de moins de 60 s au milieu d'une inactivité devient inactif (micro-réveil) ;
   - une inactivité de moins de 60 s au milieu d'un éveil redevient de l'éveil (simple pause).
4. On ne garde que les périodes actives pour calculer les descripteurs. Le taux d'inactivité
   (`nonwear_percentage`, `nonwear_duration_hours`) est conservé comme variable de contexte :
   un patient qui porte peu le capteur produit des descripteurs moins fiables.

> On ne distingue pas le sommeil du non-port : les deux sont « immobiles ». Les colonnes
> s'appellent `nonwear_*` pour des raisons historiques, mais elles mesurent l'inactivité
> totale. Une approche plus fine utiliserait l'angle du capteur, comme GGIR / van Hees, que
> nous avions testé au début.

### 5.3 Les 118 descripteurs

Calculés sur X, Y, Z et la norme, soit 4 signaux :

| Famille | Descripteurs | Intuition |
|---|---|---|
| **Statistiques globales** (60) | moyenne, écart-type, quantiles q00…q100 (pas de 10 %), kurtosis, skewness | Intensité et forme de la distribution des mouvements. Les quantiles des axes reflètent aussi les postures (orientation par rapport à la gravité). |
| **Utilisation** (2) | durée et % d'inactivité | Qualité de l'enregistrement et sédentarité |
| **Variabilité glissante** (16) | moyenne et écart-type des SD et variances sur des fenêtres de 1000 points (10 s) | Dynamique à court terme : mouvements réguliers ou par à-coups |
| **DFA glissante** (16) | moyenne et SD de α et du R² de la DFA, sur des fenêtres de 10 s espacées de 100 000 points | Complexité locale |
| **Entropie de Shannon glissante** (8) | moyenne et SD de l'entropie (histogramme à 10 classes) | Richesse ou prévisibilité du signal |
| **DFA globale** (16) | α, IC 95 % de α (min, max), R² sur tout le signal actif | Complexité à long terme |

**La DFA en une phrase** (*Detrended Fluctuation Analysis*) : on intègre le signal, on le
découpe en fenêtres de taille n, on retire la tendance linéaire de chaque fenêtre et on
mesure la fluctuation résiduelle F(n). Si F(n) ∝ n^α, l'exposant α décrit les corrélations
à long terme :

- α ≈ 0,5 : bruit blanc, pas de mémoire ;
- α ≈ 1 : bruit en 1/f, typique des systèmes physiologiques sains et adaptatifs ;
- α ≈ 1,5 : marche aléatoire, très lisse et prévisible.

L'hypothèse physiologique est qu'un patient kinésiophobe aurait un mouvement plus
contraint, donc une complexité modifiée. Le R² de l'ajustement log-log indique si la loi
de puissance est crédible.

**L'entropie de Shannon** : H = −Σ pᵢ log pᵢ sur l'histogramme des valeurs. Elle est élevée
quand le signal explore beaucoup d'amplitudes différentes et faible quand il est monotone.

## 6. Étape 2 : exploration et modélisation

Fichier : `02_modelisation.R`.

### 6.1 Préparation

- On supprime les colonnes non numériques, avec des valeurs manquantes ou constantes. La
  DFA globale de l'axe Z échoue pour un patient (NA), ce qui retire 4 colonnes : il reste
  **114 descripteurs accéléro + 5 cliniques**, plus 2 variables de cluster, soit 121.
- **Normalisation min-max** pour KNN, SVM, ACP et clustering, qui sont sensibles aux
  échelles. Les arbres et le RF n'en ont pas besoin : ils ne dépendent que de l'ordre des
  valeurs.

### 6.2 Analyse descriptive

- **Spearman** entre chaque descripteur et TAMPA. Le maximum est |ρ| ≈ 0,29 (Y_q40) :
  aucune variable seule n'est prédictive, d'où l'intérêt d'une approche multivariée.
  Spearman plutôt que Pearson, car il est robuste aux outliers et capte les relations
  monotones non linéaires.
- **ACP** : PC1 ≈ 20 % et PC2 ≈ 14 % de la variance. Aucun gradient TAMPA n'est visible.
  En revanche, les patients avec beaucoup d'inactivité ressortent comme atypiques.
- **Clustering** K-means et Ward (K = 3, choisi par la silhouette). Les étiquettes de
  cluster sont ajoutées comme variables. Une première exploration à K = 4 isolait un
  patient atypique (C01P016) dans son propre cluster.

### 6.3 Modèles comparés

| Modèle | Principe | Pourquoi l'avoir testé |
|---|---|---|
| **Dummy** | Prédit la moyenne du train | Baseline obligatoire : battre la moyenne est le minimum |
| **KNN** | Moyenne des K voisins les plus proches | Simple, non paramétrique |
| **SVR** (e1071) | Régression à marge ε, noyaux radial, polynomial, sigmoïde | Bon en grande dimension avec peu d'exemples |
| **Arbre (CART, rpart)** | Découpages binaires successifs | Interprétable |
| **Random Forest** | Moyenne de nombreux arbres sur échantillons bootstrap et sous-ensembles de variables (mtry) | Robuste, gère p > n, donne l'importance des variables |

### 6.4 Protocole d'évaluation

- **Leave-One-Out (LOO)** : 66 entraînements, chacun sur 65 patients et testé sur le
  patient restant. C'est adapté aux petits effectifs : on utilise un maximum de données et
  l'estimation ne dépend pas d'un découpage aléatoire. La variance de l'estimateur est
  en revanche élevée.
- **Métrique : MAE** (erreur absolue moyenne), dans l'unité du score : « on se trompe en
  moyenne de X points TAMPA ». Elle est plus robuste aux outliers que la RMSE.
- Dans la boucle de comparaison (`run_loo`), la normalisation est recalculée **dans
  chaque fold** sur les 65 patients d'entraînement.
- **3 configurations** : accéléro seul, clinique seul, et combiné (avec clusters).
- **Calibration** par grid search en LOO : K de 1 à 30 ; SVM avec 3 noyaux × 4 γ × 4 C ;
  arbre avec cp × maxdepth × minsplit ; RF avec ntree × mtry.
- **Réduction de dimension** : ACP (5 à 50 composantes) ou sélection des 40 variables les
  plus importantes (%IncMSE moyenné sur 30 forêts).

### 6.5 Interprétation

- **Importance %IncMSE** : augmentation de l'erreur quand on permute aléatoirement une
  variable. Elle est moyennée sur 30 forêts pour stabiliser le classement.
- **Analyse des erreurs** : prédit vs réel, puis patients avec une erreur supérieure à
  10 points, analysés par tests de Wilcoxon, test de Fisher pour le sexe, et un arbre de
  classification « mal prédit / bien prédit ».

## 7. Résultats

### Comparaison des modèles (LOO, hyperparamètres par défaut), MAE

| Modèle | Accéléro seul | Clinique seul | Combiné |
|---|---|---|---|
| Dummy (moyenne) | 6,41 | 6,41 | 6,41 |
| KNN (K = 5) | 6,70 | 7,22 | 6,77 |
| SVM (radial) | 6,38 | 6,65 | 6,43 |
| Arbre (cp = 0) | 8,35 | 7,12 | 9,48 |
| **Random Forest** | **6,39** | **5,95** | **6,37** |

Ces valeurs ont été reproduites en relançant `02_modelisation.R` sur les données du rapport.

**Lecture** :

- l'arbre seul sur-apprend fortement (pire que la baseline) ;
- le RF est le modèle le plus stable ;
- l'accélérométrie seule ne bat pas vraiment la baseline ;
- les 5 variables cliniques font mieux que les 114 descripteurs accéléro.

### Avec sélection des 40 meilleures variables (chiffres du rapport, **optimistes**, voir §8)

| Configuration | MAE RF top 40 | Gain vs dummy |
|---|---|---|
| Accéléro | 5,83 | 9,1 % |
| Clinique | 6,09 | 5 % |
| Combiné | 5,69 | 11,2 % |

### Ce qu'il faut retenir

- **L'IMC est la variable la plus importante.** Deux lectures possibles : une lecture
  clinique (la corpulence est un facteur de chronicisation et d'évitement) et une lecture
  physique (le tissu adipeux amortit les hautes fréquences, ce qui modifie l'écart-type,
  la kurtosis, etc.). Une partie du « signal accéléro » reflète donc la corpulence.
- Parmi les descripteurs accéléro, ce sont surtout la **complexité** (α et R² de la DFA,
  axe Y) et la **variabilité glissante** qui ressortent, plutôt que l'intensité brute.
- **Régression vers la moyenne** : le modèle prédit presque toujours entre 40 et 50. Les
  patients extrêmes (TAMPA ≤ 35 ou ≥ 55) sont les plus mal prédits. Avec une cible
  bruitée et peu de signal, c'est le comportement attendu. L'arbre des erreurs le
  confirme : le TAMPA réel est la variable qui discrimine le plus les mal prédits.
- Les mal prédits ne se distinguent sur aucune variable clinique prise isolément
  (Wilcoxon, p > 0,1 partout).
- **Arbre clinique** : si IMC < 21, TAMPA prédit = 36 (n = 8), sinon 46 (n = 58).
  C'est une règle lisible, mais très grossière.

## 8. Limites, dont le data leakage

### 8.1 Fuites de données identifiées

Le principe : toute étape qui « apprend » quelque chose des données (normalisation,
sélection, réduction, choix d'hyperparamètres) doit être ajustée **uniquement sur le
train de chaque fold**. Sinon, le patient de test influence son propre modèle et la
performance est surestimée.

| # | Étape | Problème | Gravité |
|---|---|---|---|
| 1 | **Sélection top 40** | L'importance %IncMSE est calculée sur les 66 patients **avec leur TAMPA**, puis le top 40 est évalué en LOO sur ces mêmes patients. Le patient de test a participé au choix des variables. | **Forte.** Les MAE 5,83 et 5,69 sont surestimées. |
| 2 | **Calibration sans CV imbriquée** | La même MAE LOO sert à choisir les hyperparamètres parmi des dizaines de combinaisons, puis à annoncer la performance. On retient le meilleur score, qui est en partie dû au hasard. | Moyenne |
| 3 | **Normalisation globale dans les calibrations** | Les grilles KNN, SVM, arbre et RF utilisent des données normalisées sur les 66 patients (min/max du test inclus). | Faible |
| 4 | **Clustering et ACP sur toute la cohorte** | Les étiquettes de cluster et les composantes sont calculées avant le LOO. C'est non supervisé (TAMPA non utilisé), mais le test influence quand même les centres et les axes. | Faible |

**Estimation la plus fiable** : le RF sur tous les descripteurs **sans sélection** et avec
les hyperparamètres par défaut (MAE 6,39 accéléro, 5,95 clinique, 6,37 combiné). Ce modèle
n'est concerné que par la fuite n° 4.

**Correctif** : une **validation croisée imbriquée** (*nested CV*). La boucle externe (LOO)
garde un patient de côté. Sur les 65 autres, la boucle interne (par exemple un 5-fold)
normalise, fait le clustering, calcule l'importance, choisit le top k et les
hyperparamètres. On n'évalue qu'ensuite sur le patient externe. En R, `caret` ou
`tidymodels` (`nested_cv`) gèrent cela proprement, à condition d'intégrer la sélection
dans le pipeline (recipe).

### 8.2 Autres limites

- **n = 66 pour environ 120 variables** : rapport très défavorable, avec un risque de
  sur-apprentissage et des estimations instables (un seul patient change sensiblement la
  MAE).
- **Cible bruitée** : TAMPA est un questionnaire déclaratif, rempli à un instant donné,
  alors que l'enregistrement couvre une semaine.
- **Confusion IMC / signal** : la corpulence modifie le signal mécanique. Il faudrait
  normaliser les descripteurs par la masse ou stratifier par IMC.
- **Fenêtre d'une semaine entière** : elle mélange sédentarité, activité et nuits, ce qui
  dilue le signal.
- **Détection d'inactivité simple** : sommeil et non-port sont confondus, et le seuil
  0,01 g a été fixé à la main.
- **Bug corrigé a posteriori** : dans le lissage RLE, la deuxième étape annulait la
  première, car les segments venant d'être passés à 1 étaient tous courts et repassaient
  à 0. Le correctif (`rle(inverse.rle(...))` entre les deux étapes) est dans le code,
  mais les descripteurs du rapport ont été calculés avant. Il faudrait relancer
  l'extraction sur les `.gt3x`.
- **Instabilité des calibrations** : en relançant le script (octobre 2026), les MAE de
  comparaison et du top 40 sont identiques au rapport (6,39 / 5,95 / 6,37 et
  5,83 / 6,09 / 5,69). En revanche, les hyperparamètres « optimaux » changent : K = 9 au
  lieu de 13, arbre avec minsplit = 10 au lieu de 2, RF avec ntree = 10 et mtry = 11 au
  lieu de 50 et 2, et l'ACP est meilleure à 5 composantes (6,05) plutôt qu'à 20. Les
  écarts de MAE entre combinaisons sont plus petits que le bruit du LOO sur 66 patients :
  la grille choisit en partie au hasard. C'est un argument de plus pour la validation
  imbriquée.

## 9. Ce que je referais autrement

1. **Validation croisée imbriquée** dès le départ, avec un pipeline `tidymodels` ou `caret`
   où toutes les transformations sont dans le fold.
2. **Moins de descripteurs mais mieux justifiés** : nombre de pas, temps passé dans chaque
   intensité (seuils de Freedson / Hildebrand en mg), fragmentation de l'activité
   (longueur des épisodes), rythme circadien (amplitude, stabilité inter-jour IS,
   variabilité intra-jour IV). Ces descripteurs sont plus interprétables et plus proches
   du concept d'« évitement ».
3. **Modèles régularisés** (Lasso, Elastic Net), adaptés à p ≈ n et qui font une
   sélection de variables intégrée et honnête si on les calibre dans la CV interne.
4. **Reformuler en classification** (TAMPA au-dessus ou en dessous de 37–40), plus utile
   cliniquement, et évaluer en AUC.
5. **Ajuster sur l'IMC** pour séparer l'effet comportemental de l'effet mécanique.
6. **Intervalles de confiance** sur la MAE (bootstrap) et test de la différence avec la
   baseline. Avec n = 66, un écart de 0,02 n'est pas significatif.
7. **Détection d'inactivité validée** (GGIR, algorithme de van Hees), avec distinction
   entre sommeil et non-port.

## 10. Questions probables et réponses

**Q : Pourquoi le Leave-One-Out plutôt qu'un train/test classique ?**
Avec 66 patients, un split 80/20 laisse 13 patients en test : l'estimation dépendrait
énormément du tirage. Le LOO utilise chaque patient comme test une fois et entraîne sur
65 patients. Ses inconvénients : un coût de calcul élevé (66 modèles par configuration)
et un estimateur à forte variance. Un 5-fold ou 10-fold répété aurait été une bonne
alternative.

**Q : Pourquoi la MAE plutôt que la RMSE ou le R² ?**
La MAE s'interprète directement en points TAMPA et pénalise moins les gros écarts sur les
patients extrêmes. Le R² sur un petit effectif est instable, et souvent négatif ici
puisqu'on bat à peine la moyenne.

**Q : Votre modèle fait 6,39 contre 6,41 pour la baseline. Ça sert à quelque chose ?**
Non, pas en l'état pour l'accéléro seul. C'est un résultat négatif, et il est informatif :
les descripteurs génériques d'activité ne capturent pas la kinésiophobie. La bonne réponse
scientifique est de le dire, puis de proposer des descripteurs plus ciblés (§9). Les
chiffres plus flatteurs du top 40 sont biaisés par une fuite de données.

**Q : Qu'est-ce que le data leakage, et où en aviez-vous ?**
C'est le fait que de l'information du jeu de test influence l'entraînement, ce qui
surestime la performance. Chez nous, la fuite principale vient de la sélection des
40 variables, faite avec l'importance RF calculée sur tous les patients avant le LOO
(détail au §8).

**Q : Comment savez-vous que la sélection top 40 est biaisée ?**
On a choisi les 40 variables qui marchent le mieux *sur ces 66 patients*. Même avec un
bruit pur, sélectionner a posteriori les variables les plus corrélées à la cible produit
une performance apparente meilleure que le hasard. Le test standard consiste à refaire
l'analyse avec la cible permutée aléatoirement : si la MAE « top 40 » bat encore la
baseline, c'est la preuve de la fuite.

**Q : Pourquoi ne pas avoir mis les EVA et l'Oswestry dans le modèle ?**
Ce sont des questionnaires remplis au même moment et très corrélés au TAMPA. Le modèle
serait meilleur, mais il répondrait à une autre question (« un questionnaire prédit-il un
autre questionnaire ? »). Ces scores ne servent que pour l'analyse descriptive des
erreurs.

**Q : Pourquoi ajouter les clusters comme variables ?**
L'idée est que des profils de comportement global (sédentaire, actif, irrégulier) peuvent
porter une information que les arbres ont du mal à reconstruire seuls. En pratique, leur
importance est faible, et ils créent une petite fuite (§8).

**Q : Pourquoi normaliser pour KNN et SVM, et pas pour RF et l'arbre ?**
KNN et SVM (noyau radial) reposent sur des distances : une variable en g² écraserait une
variable en %. Les arbres coupent sur des seuils d'une seule variable à la fois, donc
toute transformation monotone ne change rien.

**Q : Pourquoi le min-max et pas le z-score ?**
C'était le choix du cours. Le z-score serait plus robuste ici, car le min-max est sensible
aux outliers et on a des patients atypiques (fort taux d'inactivité). Pour un patient de
test, le min-max peut aussi donner des valeurs hors de [0, 1], ce qui reste acceptable.

**Q : Que mesure %IncMSE ?**
Pour chaque arbre, on permute les valeurs de la variable dans l'échantillon out-of-bag et
on mesure l'augmentation de l'erreur quadratique. Si l'erreur augmente beaucoup, la
variable est utile. C'est plus fiable que l'importance par réduction d'impureté, qui
favorise les variables continues avec beaucoup de valeurs distinctes.

**Q : Qu'est-ce que la DFA, et pourquoi l'utiliser sur un accéléromètre ?**
Voir §5.3. Elle mesure les corrélations à long terme d'un signal non stationnaire. En
physiologie (marche, rythme cardiaque), α proche de 1 traduit un système adaptatif, et
une perte de complexité est associée à la pathologie ou à la rigidité. L'hypothèse était
qu'un mouvement « contraint par la peur » serait moins complexe.

**Q : Pourquoi décimer à 1 Hz ? Ne perd-on pas de l'information ?**
La décimation ne sert qu'à **détecter l'inactivité** sur des fenêtres de 5 minutes, et la
seconde suffit pour cela. Les descripteurs sont calculés sur le signal à 100 Hz. Une
moyenne par seconde, plutôt que de prendre un point sur 100, aurait été plus propre
(anti-repliement).

**Q : Comment avez-vous choisi K = 3 pour le clustering ?**
Par le critère de la silhouette, en vérifiant visuellement sur l'ACP. Les clusters
restent peu séparés, ce qui est cohérent avec l'absence de structure claire.

**Q : Pourquoi l'arbre seul est-il si mauvais (8,35) ?**
Avec cp = 0 et minsplit = 2, il pousse jusqu'à mémoriser chaque patient : c'est du
sur-apprentissage pur, avec une variance énorme. La forêt corrige cela en moyennant des
centaines d'arbres décorrélés (bagging et mtry).

**Q : La calibration du RF a donné ntree = 10. Ce n'est pas bizarre ?**
Si. Plus d'arbres ne fait jamais sur-apprendre une forêt : ça réduit seulement la
variance. Un optimum à 10 arbres signifie que les différences de MAE entre combinaisons
sont du bruit, et qu'on a « gagné » un tirage favorable. Sur une seconde exécution,
l'optimum était à 50. La bonne lecture : on prend une valeur standard (500) et on ne
calibre que mtry, dans une CV imbriquée.

**Q : Qu'est-ce qui limite le plus le projet ?**
La taille de l'échantillon et la nature de la cible. Avec 66 patients et un score
subjectif, même un très bon descripteur aurait du mal à émerger. La confusion avec
l'IMC complique en plus l'interprétation.

**Q : Quel a été votre rôle ?**
*À compléter par moi.* Par exemple : extraction des descripteurs, détection d'inactivité,
pipeline ML, analyse des erreurs, rédaction du rapport.

**Q : Quels outils ?**
R : `read.gt3x` pour lire les fichiers ActiGraph, `RcppRoll` pour les fenêtres glissantes
rapides, `nonlinearTseries` (DFA), `entropy`, `moments`, `e1071` (SVM), `FNN` (KNN),
`rpart`, `randomForest`, `FactoMineR` et `factoextra` (ACP, clustering), `ggplot2`.
Python (`python-docx`) pour générer le rapport.

## 11. Glossaire

| Terme | Définition |
|---|---|
| **Kinésiophobie** | Peur excessive du mouvement, par crainte de douleur ou de blessure |
| **TAMPA (TSK-17)** | Questionnaire de 17 items mesurant la kinésiophobie (17 à 68) |
| **Oswestry (ODI)** | Score d'incapacité fonctionnelle liée au mal de dos |
| **EVA** | Échelle visuelle analogique de la douleur |
| **`.gt3x`** | Format binaire des accéléromètres ActiGraph |
| **g** | Unité d'accélération (1 g ≈ 9,81 m/s²) |
| **RLE** | Run-length encoding : compresse une séquence en (valeur, longueur de répétition) |
| **DFA** | Detrended Fluctuation Analysis : exposant α de corrélation à long terme |
| **LOO** | Leave-One-Out : validation croisée avec un seul individu en test à chaque fois |
| **MAE** | Mean Absolute Error : moyenne des valeurs absolues des erreurs |
| **%IncMSE** | Importance RF par permutation (hausse de la MSE quand on permute la variable) |
| **Data leakage** | Fuite d'information du test vers l'entraînement, qui surestime la performance |
| **Nested CV** | Validation croisée imbriquée : une boucle interne pour les choix, une externe pour l'évaluation |
| **Silhouette** | Mesure de séparation des clusters (−1 à 1) |
