# Données (non versionnées)

Les données proviennent d'une étude clinique sur des patients lombalgiques. Ce sont des
données de santé : elles ne sont **pas** publiées dans ce dépôt (voir `.gitignore`).

Pour exécuter le projet, placer ici :

| Chemin | Contenu | Produit par |
|---|---|---|
| `data/gt3x/*.gt3x` | Enregistrements bruts ActiGraph (tri-axial, 100 Hz, ~1 semaine), un fichier par patient, nommé `<sid>.gt3x` | Capteur |
| `data/matrice_clinique.csv` | Export clinique, séparateur `;`, une ligne par patient | Base de l'étude (REDCap) |
| `data/matrice_descripteurs.csv` | 1 ligne par patient, colonne `patient` (`<sid>.gt3x`) + 118 descripteurs | `01_extraction_descripteurs.Rmd` |

Colonnes de `matrice_clinique.csv` utilisées par `02_modelisation.R` :

| Colonne | Rôle |
|---|---|
| `sid` | Identifiant patient (ex. `C01P001`), clé de jointure avec `patient` sans `.gt3x` |
| `age`, `sexe`, `poids`, `taille`, `imc` | Variables explicatives cliniques |
| `evanuit`, `evados`, `evajamb`, `score_oswestry` | Utilisées uniquement en analyse descriptive des erreurs (exclues des features) |
| `score_tampa` | Variable cible (17–68). Les patients sans score sont exclus |
