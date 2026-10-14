# ============================================================
# Étape 2 du projet : modélisation du score TAMPA
# Entrées  : data/matrice_descripteurs.csv (sortie de 01_extraction_descripteurs.Rmd)
#            data/matrice_clinique.csv
# Sorties  : figures PDF et tableaux CSV dans outputs/
# À lancer depuis la racine du projet : Rscript 02_modelisation.R
# ============================================================

# ============================================================
# LIBRAIRIES
# ============================================================
library(dplyr)
library(e1071)
library(FNN)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ggplot2)
library(FactoMineR)
library(factoextra)

rm(list = ls())
# ============================================================
# CHARGEMENT ET PREPARATION DES DONNEES
# ============================================================

chemin_accel    <- "data/matrice_descripteurs.csv"
chemin_clinique <- "data/matrice_clinique.csv"

# Toutes les figures et tables sont écrites dans outputs/
dir.create("outputs", showWarnings = FALSE)
out <- function(fichier) file.path("outputs", fichier)

data_accel <- read.csv(chemin_accel, stringsAsFactors = FALSE)
df_raw     <- read.csv(chemin_clinique, sep = ";", stringsAsFactors = FALSE)

# Sélection des colonnes cliniques utiles et filtrage des patients sans score Tampa
cols_utiles <- c("sid", "age", "sexe", "poids", "taille", "imc",
                 "evanuit", "evados", "evajamb", "score_oswestry", "score_tampa")
df_clin <- df_raw[, cols_utiles]
df_clin$score_tampa <- suppressWarnings(as.numeric(df_clin$score_tampa))
df_clin <- df_clin[!is.na(df_clin$score_tampa), ]

print(paste("Patients avec score Tampa valide :", nrow(df_clin)))

# Fusion clinique + accélérométrie sur l'identifiant patient (inner join)
data_accel$sid  <- gsub("\\.gt3x$", "", data_accel$patient)
data_accel$patient <- NULL
df_merged <- merge(df_clin, data_accel, by = "sid")

print(paste("Patients après fusion :", nrow(df_merged)))

# Variable cible : score Tampa (kinésiophobie, échelle 17-68)
label <- df_merged$score_tampa


# ============================================================
# DEFINITION DES GROUPES DE COLONNES
# ============================================================

# Features cliniques : caractéristiques du patient (âge, sexe, morphologie)
cols_clin_feat  <- c("age", "sexe", "poids", "taille", "imc")
cols_meta       <- c("sid", "score_tampa")
# Features accélérométriques : tout ce qui reste après exclusion des colonnes non-features
# evanuit/evados/evajamb/oswestry exclus pour éviter le leakage (corrélés à Tampa)
cols_accel_feat <- setdiff(colnames(df_merged),
                           c(cols_meta, cols_clin_feat,
                             "evanuit", "evados", "evajamb", "score_oswestry"))


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

# Sélectionne les colonnes demandées et supprime celles non numériques,
# avec NA, ou sans variance (constantes)
prepare <- function(df, cols) {
  X <- df[, intersect(cols, colnames(df)), drop = FALSE]
  X <- X[, sapply(X, is.numeric), drop = FALSE]
  X <- X[, colSums(is.na(X)) == 0, drop = FALSE]
  X <- X[, apply(X, 2, stats::var) > 1e-10, drop = FALSE]
  return(X)
}

# Normalisation min-max colonne par colonne → valeurs dans [0, 1]
# Utilisée pour l'ACP et le clustering (sensibles aux échelles)
# ATTENTION : appliquée ici sur tous les patients → ne pas utiliser directement dans run_loo
normalize <- function(X) {
  as.data.frame(lapply(X, function(col) {
    rng <- max(col) - min(col)
    if (rng == 0) return(rep(0, length(col)))
    (col - min(col)) / rng
  }))
}


# ============================================================
# CONSTRUCTION DES MATRICES DE FEATURES
# ============================================================

# Versions normalisées : utilisées pour l'ACP et le clustering (sensibles aux échelles)
feat_accel <- normalize(prepare(df_merged, cols_accel_feat))
feat_clin  <- normalize(prepare(df_merged, cols_clin_feat))
feat_both  <- normalize(prepare(df_merged, c(cols_accel_feat, cols_clin_feat)))

# Versions brutes : utilisées dans run_loo
# La normalisation est appliquée par fold dans la boucle pour KNN et SVM uniquement
# (arbre et forêt sont invariants à l'échelle)
feat_accel_raw <- prepare(df_merged, cols_accel_feat)
feat_clin_raw  <- prepare(df_merged, cols_clin_feat)
feat_both_raw  <- prepare(df_merged, c(cols_accel_feat, cols_clin_feat))

print(paste("Features accel   :", ncol(feat_accel)))
print(paste("Features clinique:", ncol(feat_clin)))
print(paste("Features combiné :", ncol(feat_both)))


# ============================================================
# EXPLORATION : DISTRIBUTIONS DES VARIABLES CLINIQUES
# ============================================================

pdf(out("1_distributions_variables.pdf"), width = 10, height = 6)
par(mfrow = c(2, 3))
boxplot(df_merged$age,            main = "Age",                   col = "steelblue")
boxplot(df_merged$imc,            main = "IMC",                   col = "steelblue")
boxplot(df_merged$score_oswestry, main = "Oswestry (info seule)", col = "tomato")
boxplot(df_merged$score_tampa,    main = "TAMPA (cible)",         col = "darkgreen")
boxplot(df_merged$evanuit,        main = "EVA nuit",              col = "orange")
boxplot(df_merged$evados,         main = "EVA dos",               col = "orange")
dev.off()

# Distribution détaillée du score Tampa : histogramme + densité + moyenne/médiane
p_dist_tampa <- ggplot(df_merged, aes(x = score_tampa)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 2,
                 fill = "darkgreen", color = "white", alpha = 0.75) +
  geom_density(color = "black", linewidth = 1) +
  geom_vline(aes(xintercept = mean(score_tampa, na.rm = TRUE)),
             color = "red", linetype = "dashed", linewidth = 0.9) +
  geom_vline(aes(xintercept = median(score_tampa, na.rm = TRUE)),
             color = "orange", linetype = "dotted", linewidth = 0.9) +
  annotate("text",
           x = mean(df_merged$score_tampa, na.rm = TRUE) + 0.5,
           y = Inf, vjust = 1.5, hjust = 0,
           label = paste0("Moyenne = ", round(mean(df_merged$score_tampa, na.rm = TRUE), 1)),
           color = "red", size = 3.5) +
  annotate("text",
           x = median(df_merged$score_tampa, na.rm = TRUE) + 0.5,
           y = Inf, vjust = 3, hjust = 0,
           label = paste0("Médiane = ", round(median(df_merged$score_tampa, na.rm = TRUE), 1)),
           color = "orange", size = 3.5) +
  labs(title = "Distribution du score TAMPA",
       subtitle = paste0("n = ", sum(!is.na(df_merged$score_tampa)),
                         "  |  Min = ", min(df_merged$score_tampa, na.rm = TRUE),
                         "  Max = ", max(df_merged$score_tampa, na.rm = TRUE),
                         "  |  SD = ", round(sd(df_merged$score_tampa, na.rm = TRUE), 1)),
       x = "Score TAMPA", y = "Densité") +
  theme_minimal(base_size = 13)

pdf(out("2_distribution_score_tampa.pdf"), width = 8, height = 6)
print(p_dist_tampa)
dev.off()


# ============================================================
# CORRELATIONS UNIVARIEES DE SPEARMAN (descripteurs accel vs TAMPA)
# ============================================================

rho_spearman <- sapply(feat_accel_raw, function(col)
  cor(col, label, method = "spearman"))
spearman_df <- data.frame(variable = names(rho_spearman),
                          rho      = round(rho_spearman, 3))
spearman_df <- spearman_df[order(-abs(spearman_df$rho)), ]

print("Top 10 corrélations de Spearman avec le score TAMPA :")
print(head(spearman_df, 10), row.names = FALSE)
write.csv(spearman_df, out("correlations_spearman.csv"), row.names = FALSE)


# ============================================================
# ANALYSE EN COMPOSANTES PRINCIPALES (ACP)
# ============================================================

# ACP sur les features combinées normalisées pour visualiser la structure des données
data_pour_acp <- cbind(feat_both, score_tampa = label)
pca_result <- PCA(data_pour_acp[, -ncol(data_pour_acp)],
                  graph = FALSE, ncp = min(120, ncol(feat_both)))

# Projection des individus sur les 2 premières composantes, colorés par score Tampa
acp_coords <- as.data.frame(pca_result$ind$coord[, 1:2])
colnames(acp_coords) <- c("Dim1", "Dim2")
acp_coords$score_tampa <- label
acp_coords$nonwear_pct <- df_merged$nonwear_percentage

p_acp_tampa <- ggplot(acp_coords, aes(x = Dim1, y = Dim2, color = score_tampa)) +
  geom_point(size = 3) +
  scale_color_gradient(low = "green", high = "red", name = "Score TAMPA") +
  labs(title = "ACP - Individus colorés par score TAMPA",
       x = "Dim 1", y = "Dim 2") +
  theme_minimal()

pdf(out("3_acp_individus_tampa.pdf"), width = 8, height = 6)
print(p_acp_tampa)
dev.off()

acp_coords$sid <- df_merged$sid
top5_nonwear <- acp_coords[order(-acp_coords$nonwear_pct), ][1:5, ]

p_acp_nonwear <- ggplot(acp_coords, aes(x = Dim1, y = Dim2, color = nonwear_pct)) +
  geom_point(size = 3) +
  geom_text(data = top5_nonwear, aes(label = sid), color = "black",
            size = 3, nudge_y = 0.3, fontface = "bold") +
  scale_color_gradient(low = "steelblue", high = "orange", name = "Non-port (%)") +
  labs(title = "ACP - Individus colorés par taux de non-port du capteur",
       x = "Dim 1", y = "Dim 2") +
  theme_minimal()

pdf(out("4_acp_individus_nonwear.pdf"), width = 8, height = 6)
print(p_acp_nonwear)
dev.off()

pdf(out("5_acp_variables.pdf"), width = 10, height = 10)
print(fviz_pca_var(pca_result, col.var = "contrib",
             gradient.cols = c("blue", "yellow", "red"),
             repel = TRUE, title = "ACP - Cercle des corrélations"))
dev.off()


# ============================================================
# CLUSTERING
# ============================================================

# K-means (K=3) et clustering hiérarchique (Ward) sur features combinées normalisées
# Les labels de clusters seront ajoutés comme features supplémentaires dans le ML
# LIMITE (fuite faible) : clustering calculé sur les 66 patients avant le LOO,
# le patient de test contribue donc aux centres. Non supervisé (TAMPA non utilisé).
set.seed(42)
km3      <- kmeans(feat_both, centers = 3, nstart = 25)
dist_mat <- dist(feat_both, method = "euclidean")
hclust3  <- hclust(dist_mat, method = "ward.D2")

cluster_kmeans_viz <- as.factor(km3$cluster)
cluster_hclust_viz <- as.factor(cutree(hclust3, k = 3))
cluster_kmeans_num <- as.integer(km3$cluster)
cluster_hclust_num <- as.integer(cutree(hclust3, k = 3))

# Choix du nombre de clusters optimal par la méthode silhouette
pdf(out("6_clustering_choix_k.pdf"), width = 8, height = 5)
print(fviz_nbclust(feat_both, kmeans, method = "silhouette", k.max = 8) +
  labs(title = "Choix du nombre de clusters K-means (silhouette)"))
dev.off()

pdf(out("7_clustering_kmeans_k3.pdf"), width = 8, height = 6)
print(fviz_pca_ind(pca_result,
             habillage   = cluster_kmeans_viz,
             palette     = "jco",
             addEllipses = TRUE, ellipse.level = 0.95,
             title = "K-means K=3 projeté sur ACP"))
dev.off()

pdf(out("8_clustering_hclust_k3.pdf"), width = 8, height = 6)
print(fviz_pca_ind(pca_result,
             habillage   = cluster_hclust_viz,
             palette     = "Dark2",
             addEllipses = TRUE, ellipse.level = 0.95,
             title = "Clustering hiérarchique K=3 projeté sur ACP"))
dev.off()

# Ajout des labels de clusters comme features supplémentaires
feat_both_clust <- feat_both
feat_both_clust$cluster_kmeans <- cluster_kmeans_num
feat_both_clust$cluster_hclust <- cluster_hclust_num

# Version brute pour run_loo (normalisation faite par fold à l'intérieur)
feat_both_clust_raw <- feat_both_raw
feat_both_clust_raw$cluster_kmeans <- cluster_kmeans_num
feat_both_clust_raw$cluster_hclust <- cluster_hclust_num

print(paste("Features combiné + clusters :", ncol(feat_both_clust)))


# ============================================================
# FONCTION DE VALIDATION CROISEE LOO
# ============================================================

# Leave-One-Out : pour chaque patient i, entraîne sur les n-1 autres et prédit i
# Reçoit des features brutes (non normalisées)
# - KNN et SVM : normalisation min-max par fold sur le train, appliquée au test
# - Arbre et Random Forest : features brutes (invariants à l'échelle)
# - Hyperparamètres par défaut ici → calibration faite séparément
run_loo <- function(features, label, config_name, k_knn = 5) {
  n <- nrow(features)
  mae_mat <- matrix(NA, nrow = n, ncol = 5,
                    dimnames = list(NULL, c("dummy","knn","svm","tree","forest")))

  print(paste("LOO :", config_name, "|", n, "patients"))
  t0 <- Sys.time()

  for (i in 1:n) {
    test_x  <- features[i, , drop = FALSE]
    train_x <- features[-i, , drop = FALSE]
    test_y  <- label[i]
    train_y <- label[-i]

    # Normalisation par fold pour KNN et SVM uniquement
    # Le min/max est calculé sur le train seul → pas de data leakage
    mins <- apply(train_x, 2, min)
    rng  <- apply(train_x, 2, max) - mins
    rng[rng == 0] <- 1
    train_x_norm <- sweep(sweep(train_x, 2, mins, "-"), 2, rng, "/")
    test_x_norm  <- sweep(sweep(test_x,  2, mins, "-"), 2, rng, "/")

    train_df      <- data.frame(train_x,      score_tampa = train_y)
    test_df       <- data.frame(test_x,       score_tampa = test_y)
    train_df_norm <- data.frame(train_x_norm, score_tampa = train_y)
    test_df_norm  <- data.frame(test_x_norm,  score_tampa = test_y)

    # Baseline : prédit la moyenne du train (référence naïve)
    dummy_pred <- mean(train_y)

    # KNN (normalisé par fold)
    k_eff    <- min(k_knn, nrow(train_x_norm))
    knn_pred <- knn.reg(train_x_norm, test_x_norm, train_y, k = k_eff)$pred
    
    svm_mod  <- svm(score_tampa ~ ., data = train_df_norm)
    svm_pred <- as.numeric(predict(svm_mod, test_df_norm))
    
    # Arbre de décision (features brutes)
    tree_mod  <- rpart(score_tampa ~ ., data = train_df, method = "anova",
                       control = rpart.control(cp = 0, minsplit = 2))
    tree_pred <- as.numeric(predict(tree_mod, test_df))

    # Random Forest (features brutes)
    rf_mod  <- randomForest(score_tampa ~ ., data = train_df, ntree = 200)
    rf_pred <- as.numeric(predict(rf_mod, test_df))

    mae_mat[i, ] <- abs(c(dummy_pred, knn_pred, svm_pred, tree_pred, rf_pred) - test_y)
  }

  duree <- round(difftime(Sys.time(), t0, units = "secs"), 1)
  print(paste("Durée :", duree, "sec"))
  print("MAE par modèle :")
  print(round(colMeans(as.data.frame(mae_mat)), 2))

  return(as.data.frame(mae_mat))
}


# ============================================================
# LOO 1 : COMPARAISON DES 3 CONFIGURATIONS — TOUS LES MODELES
# ============================================================
# Hyperparamètres par défaut → sert à identifier le meilleur modèle
# La calibration fine est faite ensuite sur le modèle retenu (RF)

mae_accel <- run_loo(feat_accel_raw,      label, "Accel seul",              k_knn = 5)
mae_clin  <- run_loo(feat_clin_raw,       label, "Clinique seul",           k_knn = 5)
mae_both  <- run_loo(feat_both_clust_raw, label, "Accel+Clinique+Clusters", k_knn = 5)

recap <- data.frame(
  modele   = c("dummy","knn","svm","tree","forest"),
  accel    = round(colMeans(mae_accel), 2),
  clinique = round(colMeans(mae_clin),  2),
  combine  = round(colMeans(mae_both),  2)
)
print("Récapitulatif MAE moyen :")
print(recap)

pdf(out("9_comparaison_modeles.pdf"), width = 14, height = 12)
layout(matrix(c(1, 2, 3, 3), nrow = 2, byrow = TRUE), heights = c(4, 3))

df_box1 <- data.frame(
  erreur = c(mae_accel$forest, mae_clin$forest, mae_both$forest),
  config = rep(c("Accel", "Clinique", "Combiné"), each = nrow(mae_accel))
)
boxplot(erreur ~ config, data = df_box1, col = rainbow(3),
        main = "Random Forest - 3 configurations", ylab = "MAE", las = 1)
abline(h = mean(mae_both$dummy), lty = 2, col = "red", lwd = 2)
boxplot(mae_both, col = rainbow(5), las = 2,
        main = "Tous modèles - Config combinée", ylab = "MAE")
abline(h = mean(mae_both$dummy), lty = 2, col = "red", lwd = 2)

mae_dummy_accel <- mean(mae_accel$dummy)
mae_dummy_clin  <- mean(mae_clin$dummy)
mae_dummy_both  <- mean(mae_both$dummy)

recap_pct <- data.frame(
  Modele        = c("Dummy (baseline)", "KNN", "SVM", "Arbre", "Random Forest"),
  MAE_Accel     = round(colMeans(mae_accel), 2),
  Gain_Accel    = paste0(round((1 - colMeans(mae_accel) / mae_dummy_accel) * 100, 1), "%"),
  MAE_Clinique  = round(colMeans(mae_clin),  2),
  Gain_Clinique = paste0(round((1 - colMeans(mae_clin)  / mae_dummy_clin)  * 100, 1), "%"),
  MAE_Combine   = round(colMeans(mae_both),  2),
  Gain_Combine  = paste0(round((1 - colMeans(mae_both)  / mae_dummy_both)  * 100, 1), "%"),
  row.names = NULL
)

par(mar = c(1, 1, 3, 1))
plot.new()
title(main = "MAE moyen et gain vs baseline dummy (LOO)", cex.main = 1.1)
addmargins_table <- as.matrix(recap_pct)
col_headers <- colnames(recap_pct)
n_col <- length(col_headers)
n_row <- nrow(recap_pct)
x_pos    <- seq(0.02, 0.98, length.out = n_col)
y_header <- 0.88
y_rows   <- seq(0.72, 0.08, length.out = n_row)

for (j in seq_along(col_headers)) {
  text(x_pos[j], y_header, col_headers[j], font = 2, cex = 0.85, adj = 0.5)
}
abline(h = 0.82, col = "gray40", lwd = 1.2)

for (i in seq_len(n_row)) {
  if (i %% 2 == 0) rect(0, y_rows[i] - 0.045, 1, y_rows[i] + 0.045,
                        col = "#f0f0f0", border = NA)
  for (j in seq_len(n_col)) {
    txt <- addmargins_table[i, j]
    clr <- "black"
    if (grepl("Gain", col_headers[j])) {
      val <- as.numeric(sub("%", "", txt))
      clr <- ifelse(is.na(val), "black", ifelse(val > 0, "darkgreen", "red"))
    }
    text(x_pos[j], y_rows[i], txt, cex = 0.82, col = clr, adj = 0.5)
  }
}
abline(h = 0.78, col = "gray70", lwd = 0.8)
dev.off()

# PDF : boxplots accélérométriques uniquement (LOO1, tous modèles, config accel)
pdf(out("9_accel_boxplots.pdf"), width = 8, height = 6)
boxplot(mae_accel[, c("dummy", "knn", "svm", "tree", "forest")],
        col  = c("gray70", "steelblue", "orchid", "darkorange", "forestgreen"),
        names = c("Dummy", "KNN", "SVM", "Arbre", "RF"),
        main = "MAE – Features accélérométriques (LOO1)",
        ylab = "MAE", las = 1)
abline(h = mae_dummy_accel, lty = 2, col = "red", lwd = 2)
legend("topright", legend = "Baseline dummy", lty = 2, col = "red", lwd = 2, bty = "n")
dev.off()


# LIMITE : les calibrations ci-dessous utilisent feat_both_clust, normalisé sur tous les
# patients (pas par fold), et la MAE LOO sert à la fois à choisir les hyperparamètres et à
# évaluer le modèle (pas de validation croisée imbriquée) -> MAE optimiste.
n_patients <- nrow(feat_both_clust)

k_max     <- min(30, n_patients - 1)
k_grid    <- 1:k_max
mae_knn_k <- numeric(length(k_grid))

print(paste("Calibration KNN : test K de 1 à", k_max))

for (idx_k in seq_along(k_grid)) {
  k_val <- k_grid[idx_k]
  errs  <- numeric(n_patients)
  for (i in 1:n_patients) {
    pred    <- knn.reg(feat_both_clust[-i, ], feat_both_clust[i, , drop = FALSE],
                       label[-i], k = k_val)$pred
    errs[i] <- abs(pred - label[i])
  }
  mae_knn_k[idx_k] <- mean(errs)
}

k_optimal <- k_grid[which.min(mae_knn_k)]
print(paste("K optimal :", k_optimal, "| MAE =", round(min(mae_knn_k), 2)))

pdf(out("10_calibration_knn.pdf"), width = 8, height = 6)
plot(k_grid, mae_knn_k, type = "b", col = "steelblue", lwd = 2,
     main = "Calibration KNN", xlab = "K", ylab = "MAE moyen")
abline(v = k_optimal, lty = 2, col = "red", lwd = 2)
text(k_optimal + 1, max(mae_knn_k), paste("K =", k_optimal), col = "red")
dev.off()


print("Calibration SVM")

kernel_list <- c("radial", "polynomial", "sigmoid")
gamma_grid  <- c(0.001, 0.01, 0.1, 1)
cost_grid   <- c(0.1, 1, 10, 100)

best_svm_mae    <- Inf
best_svm_params <- list()
svm_results     <- data.frame()

for (kern in kernel_list) {
  for (gam in gamma_grid) {
    for (cst in cost_grid) {
      errs <- numeric(n_patients)
      for (i in 1:n_patients) {
        train_df <- data.frame(feat_both_clust[-i, ], score_tampa = label[-i])
        test_df  <- data.frame(feat_both_clust[i, , drop = FALSE], score_tampa = label[i])
        tryCatch({
          mod      <- svm(score_tampa ~ ., data = train_df,
                          kernel = kern, gamma = gam, cost = cst)
          pred     <- as.numeric(predict(mod, test_df))
          errs[i]  <- abs(pred - label[i])
        }, error = function(e) { errs[i] <<- NA })
      }
      mae_val     <- mean(errs, na.rm = TRUE)
      svm_results <- rbind(svm_results,
                           data.frame(kernel = kern, gamma = gam,
                                      cost = cst, mae = round(mae_val, 3)))
      if (!is.na(mae_val) && mae_val < best_svm_mae) {
        best_svm_mae    <- mae_val
        best_svm_params <- list(kernel = kern, gamma = gam, cost = cst)
      }
    }
  }
}

print("Meilleurs hyperparamètres SVM :")
print(best_svm_params)
print(paste("MAE optimal SVM :", round(best_svm_mae, 2)))

svm_ylim <- quantile(svm_results$mae, 0.75, na.rm = TRUE) * 2

pdf(out("11_calibration_svm.pdf"), width = 10, height = 6)
print(
  ggplot(svm_results, aes(x = factor(cost), y = mae,
                          color = factor(gamma), group = factor(gamma))) +
    geom_line() + geom_point() +
    facet_wrap(~kernel) +
    coord_cartesian(ylim = c(0, svm_ylim)) +
    labs(title = "Calibration SVM : MAE selon kernel, gamma, cost (échelle ajustée)",
         subtitle = paste0("Axe Y limité au double du 3e quartile (", round(svm_ylim, 1),
                           ") — configurations hors échelle exclues visuellement"),
         x = "Cost", y = "MAE", color = "Gamma") +
    theme_minimal()
)
dev.off()


print("Calibration Arbre de décision")

cp_grid       <- c(0, 0.001, 0.01, 0.05, 0.1)
maxdepth_grid <- c(3, 5, 10, 20, 30)
minsplit_grid <- c(2, 5, 10, 20)

best_tree_mae    <- Inf
best_tree_params <- list()
tree_results     <- data.frame()

for (cp_val in cp_grid) {
  for (md in maxdepth_grid) {
    for (ms in minsplit_grid) {
      errs <- numeric(n_patients)
      for (i in 1:n_patients) {
        train_df <- data.frame(feat_both_clust[-i, ], score_tampa = label[-i])
        test_df  <- data.frame(feat_both_clust[i, , drop = FALSE], score_tampa = label[i])
        mod  <- rpart(score_tampa ~ ., data = train_df, method = "anova",
                      control = rpart.control(cp = cp_val, maxdepth = md, minsplit = ms))
        pred <- as.numeric(predict(mod, test_df))
        errs[i] <- abs(pred - label[i])
      }
      mae_val      <- mean(errs)
      tree_results <- rbind(tree_results,
                            data.frame(cp = cp_val, maxdepth = md,
                                       minsplit = ms, mae = round(mae_val, 3)))
      if (mae_val < best_tree_mae) {
        best_tree_mae    <- mae_val
        best_tree_params <- list(cp = cp_val, maxdepth = md, minsplit = ms)
      }
    }
  }
}

print("Meilleurs hyperparamètres Arbre :")
print(best_tree_params)
print(paste("MAE optimal Arbre :", round(best_tree_mae, 2)))

pdf(out("12_calibration_arbre.pdf"), width = 10, height = 6)
tree_res_agg <- aggregate(mae ~ cp + maxdepth, data = tree_results, FUN = min)
print(ggplot(tree_res_agg, aes(x = factor(maxdepth), y = mae,
                         color = factor(cp), group = factor(cp))) +
  geom_line() + geom_point() +
  labs(title = "Calibration Arbre : MAE selon cp et maxdepth",
       x = "maxdepth", y = "MAE (min sur minsplit)", color = "cp") +
  theme_minimal())
dev.off()


print("Calibration Random Forest")

set.seed(42)
ntree_grid <- c(10, 50, 100, 200, 500, 1000, 2000)
n_feat     <- ncol(feat_both_clust)
mtry_grid  <- unique(c(2, floor(n_feat / 3), floor(sqrt(n_feat)), floor(n_feat / 2)))
mtry_grid  <- mtry_grid[mtry_grid >= 1 & mtry_grid <= n_feat]

best_rf_mae    <- Inf
best_rf_params <- list()
rf_results     <- data.frame()

for (nt in ntree_grid) {
  for (mt in mtry_grid) {
    errs <- numeric(n_patients)
    for (i in 1:n_patients) {
      train_df <- data.frame(feat_both_clust[-i, ], score_tampa = label[-i])
      test_df  <- data.frame(feat_both_clust[i, , drop = FALSE])
      mod  <- randomForest(score_tampa ~ ., data = train_df, ntree = nt, mtry = mt)
      pred <- as.numeric(predict(mod, test_df))
      errs[i] <- abs(pred - label[i])
    }
    mae_val    <- mean(errs)
    rf_results <- rbind(rf_results,
                        data.frame(ntree = nt, mtry = mt, mae = round(mae_val, 3)))
    if (mae_val < best_rf_mae) {
      best_rf_mae    <- mae_val
      best_rf_params <- list(ntree = nt, mtry = mt)
    }
  }
}

print("Meilleurs hyperparamètres Random Forest :")
print(best_rf_params)
print(paste("MAE optimal RF :", round(best_rf_mae, 2)))

pdf(out("13_calibration_random_forest.pdf"), width = 10, height = 6)
print(ggplot(rf_results, aes(x = ntree, y = mae,
                       color = factor(mtry), group = factor(mtry))) +
  geom_line() + geom_point() +
  scale_x_log10() +
  labs(title = "Calibration Random Forest : MAE selon ntree et mtry",
       x = "ntree (échelle log)", y = "MAE", color = "mtry") +
  theme_minimal())
dev.off()


# LIMITE (fuite principale) : importance calculée sur les 66 patients AVEC leur score TAMPA.
# Le top 40 qui en découle est ensuite évalué en LOO sur ces mêmes patients : le patient de
# test a participé à la sélection -> les MAE "top 40" (5,83 / 5,69) sont surestimées.
# Correctif : refaire la sélection à l'intérieur de chaque fold (LOO imbriqué).
df_full    <- data.frame(feat_both_clust, score_tampa = label)
n_iter     <- 30
imp_matrix <- matrix(NA, nrow = ncol(feat_both_clust), ncol = n_iter)
rownames(imp_matrix) <- colnames(feat_both_clust)

print(paste("Calcul importance (", n_iter, "itérations)..."))
for (i in 1:n_iter) {
  rf_temp <- randomForest(score_tampa ~ ., data = df_full, ntree = 200, importance = TRUE)
  imp_matrix[, i] <- importance(rf_temp, type = 1)[, 1]
}

imp_mean <- rowMeans(imp_matrix)
imp_df   <- data.frame(variable = names(imp_mean), importance = imp_mean)
imp_df   <- imp_df[order(-imp_df$importance), ]
imp_df$type <- ifelse(imp_df$variable %in% cols_clin_feat, "Clinique",
                      ifelse(grepl("cluster", imp_df$variable), "Cluster", "Accel"))

print("Top 30 variables :")
print(head(imp_df, 30))

pdf(out("14_importance_variables.pdf"), width = 10, height = 8)
top30 <- head(imp_df, 30)
barplot(top30$importance, names.arg = top30$variable, las = 2, cex.names = 0.6,
        col = ifelse(top30$type == "Clinique", "tomato",
                     ifelse(top30$type == "Cluster", "gold", "steelblue")),
        main = paste("Top 30 variables (moyenné sur", n_iter, "itérations)"),
        ylab = "%IncMSE")
legend("topright", legend = c("Clinique", "Accel", "Cluster"),
       fill = c("tomato", "steelblue", "gold"))
dev.off()

print("Importance moyenne par type :")
print(aggregate(importance ~ type, imp_df, mean))

# Importance recalculée sur les features accélérométriques uniquement
# → permet un ranking propre sans influence des variables cliniques/cluster
df_accel_only    <- data.frame(feat_accel_raw, score_tampa = label)
imp_matrix_accel <- matrix(NA, nrow = ncol(feat_accel_raw), ncol = n_iter)
rownames(imp_matrix_accel) <- colnames(feat_accel_raw)

print(paste("Calcul importance accel uniquement (", n_iter, "itérations)..."))
for (i in 1:n_iter) {
  rf_tmp <- randomForest(score_tampa ~ ., data = df_accel_only, ntree = 200, importance = TRUE)
  imp_matrix_accel[, i] <- importance(rf_tmp, type = 1)[, 1]
}

imp_mean_accel <- rowMeans(imp_matrix_accel)
imp_df_accel   <- data.frame(variable = names(imp_mean_accel), importance = imp_mean_accel)
imp_df_accel   <- imp_df_accel[order(-imp_df_accel$importance), ]

print("Top 40 variables accélérométriques (importance accel only) :")
print(head(imp_df_accel, 40))

pdf(out("14b_importance_accel_only.pdf"), width = 10, height = 8)
top40_accel_plot <- head(imp_df_accel, 40)
barplot(top40_accel_plot$importance, names.arg = top40_accel_plot$variable,
        las = 2, cex.names = 0.6, col = "steelblue",
        main = paste("Top 40 variables accélérométriques – importance accel only\n(moyenné sur", n_iter, "itérations)"),
        ylab = "%IncMSE")
dev.off()



# ============================================================
# RF TOP 40 FEATURES - COMPARAISON 3 CONFIGURATIONS
# ============================================================

top40_combined_vars <- as.character(head(imp_df, 40)$variable)
top40_accel_vars    <- as.character(head(imp_df_accel$variable, min(40, nrow(imp_df_accel))))
top40_clin_vars     <- as.character(imp_df$variable[imp_df$type == "Clinique"])

feat_top40_accel <- feat_accel_raw[,      intersect(top40_accel_vars,    colnames(feat_accel_raw)),      drop = FALSE]
feat_top40_clin  <- feat_clin_raw[,       intersect(top40_clin_vars,     colnames(feat_clin_raw)),       drop = FALSE]
feat_top40_comb  <- feat_both_clust_raw[, intersect(top40_combined_vars, colnames(feat_both_clust_raw)), drop = FALSE]

print(paste("Top40 accel :", ncol(feat_top40_accel), "features"))
print(paste("Top40 clin  :", ncol(feat_top40_clin),  "features"))
print(paste("Top40 comb  :", ncol(feat_top40_comb),  "features"))


# ============================================================
# CALIBRATION RF SUR TOP 40 FEATURES
# ============================================================

print("Calibration Random Forest sur top 40 features")

set.seed(42)
n_feat_top40     <- ncol(feat_top40_comb)
mtry_grid_top40  <- unique(c(2, floor(n_feat_top40 / 3), floor(sqrt(n_feat_top40)), floor(n_feat_top40 / 2)))
mtry_grid_top40  <- mtry_grid_top40[mtry_grid_top40 >= 1 & mtry_grid_top40 <= n_feat_top40]

best_rf_mae_top40    <- Inf
best_rf_params_top40 <- list()
rf_results_top40     <- data.frame()

for (nt in ntree_grid) {
  for (mt in mtry_grid_top40) {
    errs <- numeric(n_patients)
    for (i in 1:n_patients) {
      train_df <- data.frame(feat_top40_comb[-i, ], score_tampa = label[-i])
      test_df  <- data.frame(feat_top40_comb[i, , drop = FALSE])
      mod  <- randomForest(score_tampa ~ ., data = train_df, ntree = nt, mtry = mt)
      pred <- as.numeric(predict(mod, test_df))
      errs[i] <- abs(pred - label[i])
    }
    mae_val          <- mean(errs)
    rf_results_top40 <- rbind(rf_results_top40,
                              data.frame(ntree = nt, mtry = mt, mae = round(mae_val, 3)))
    if (mae_val < best_rf_mae_top40) {
      best_rf_mae_top40    <- mae_val
      best_rf_params_top40 <- list(ntree = nt, mtry = mt)
    }
  }
}

print("Meilleurs hyperparamètres RF top 40 :")
print(best_rf_params_top40)
print(paste("MAE optimal RF top 40 :", round(best_rf_mae_top40, 2)))

pdf(out("13b_calibration_rf_top40.pdf"), width = 10, height = 6)
print(ggplot(rf_results_top40, aes(x = ntree, y = mae,
                             color = factor(mtry), group = factor(mtry))) +
  geom_line() + geom_point() +
  scale_x_log10() +
  labs(title = "Calibration RF top 40 : MAE selon ntree et mtry",
       x = "ntree (échelle log)", y = "MAE", color = "mtry") +
  theme_minimal())
dev.off()


# LOO RF uniquement : plus rapide que run_loo car pas de KNN/SVM/arbre
# Utilisé pour comparer les 3 configs sur le top 40 avec les hyperparamètres recalibrés
run_loo_rf_top40 <- function(features_raw, label, ntree, mtry_base) {
  n    <- nrow(features_raw)
  errs <- numeric(n)
  mt   <- min(mtry_base, ncol(features_raw))
  for (i in 1:n) {
    train_df <- data.frame(features_raw[-i, ], score_tampa = label[-i])
    test_df  <- data.frame(features_raw[i,  , drop = FALSE])
    mod      <- randomForest(score_tampa ~ ., data = train_df, ntree = ntree, mtry = mt)
    errs[i]  <- abs(as.numeric(predict(mod, test_df)) - label[i])
  }
  return(errs)
}

# ============================================================
# LOO 2 : RF TOP 40 — 3 CONFIGURATIONS SEPAREES
# ============================================================
# Hyperparamètres issus de la calibration sur top 40 (best_rf_params_top40)
set.seed(42)
mae_rf40_accel <- run_loo_rf_top40(feat_top40_accel, label, best_rf_params_top40$ntree, best_rf_params_top40$mtry)
mae_rf40_clin  <- run_loo_rf_top40(feat_top40_clin,  label, best_rf_params_top40$ntree, best_rf_params_top40$mtry)
mae_rf40_comb  <- run_loo_rf_top40(feat_top40_comb,  label, best_rf_params_top40$ntree, best_rf_params_top40$mtry)

mae_dummy_val <- mean(mae_accel$dummy)

print("=== RF COMPLET vs RF TOP 40 ===")
print(paste("Accel    - RF:", round(mean(mae_accel$forest), 3),
            "| RF top40:", round(mean(mae_rf40_accel), 3)))
print(paste("Clinique - RF:", round(mean(mae_clin$forest), 3),
            "| RF top40:", round(mean(mae_rf40_clin), 3)))
print(paste("Combiné  - RF:", round(mean(mae_both$forest), 3),
            "| RF top40:", round(mean(mae_rf40_comb), 3)))

pdf(out("9b_comparaison_rf_top40.pdf"), width = 14, height = 12)
layout(matrix(c(1, 2, 3, 3), nrow = 2, byrow = TRUE), heights = c(4, 3))

# Plot 1 : RF complet - 3 configurations
df_box_complet <- data.frame(
  erreur = c(mae_accel$forest, mae_clin$forest, mae_both$forest),
  config = rep(c("Accel", "Clinique", "Combiné"), each = nrow(mae_accel))
)
boxplot(erreur ~ config, data = df_box_complet,
        col  = c("steelblue", "tomato", "gold"),
        main = "RF complet – 3 configurations", ylab = "MAE", las = 1)
abline(h = mae_dummy_val, lty = 2, col = "red", lwd = 2)

# Plot 2 : RF top 40 - 3 configurations
df_box_top40 <- data.frame(
  erreur = c(mae_rf40_accel, mae_rf40_clin, mae_rf40_comb),
  config = rep(c("Accel top40", "Clinique top40", "Combiné top40"),
               each = length(mae_rf40_accel))
)
boxplot(erreur ~ config, data = df_box_top40,
        col  = c("steelblue3", "tomato3", "goldenrod3"),
        main = "RF top 40 features – 3 configurations", ylab = "MAE", las = 1)
abline(h = mae_dummy_val, lty = 2, col = "red", lwd = 2)

# Table MAE + gain vs dummy
gain_pct <- function(mae_vec, dummy) {
  round((1 - mae_vec / dummy) * 100, 1)
}
recap_rf40 <- data.frame(
  Modele        = c("Dummy (baseline)", "RF complet", "RF top 40"),
  MAE_Accel     = round(c(mae_dummy_val, mean(mae_accel$forest), mean(mae_rf40_accel)), 2),
  Gain_Accel    = paste0(gain_pct(c(mae_dummy_val, mean(mae_accel$forest), mean(mae_rf40_accel)), mae_dummy_val), "%"),
  MAE_Clinique  = round(c(mae_dummy_val, mean(mae_clin$forest),  mean(mae_rf40_clin)),  2),
  Gain_Clinique = paste0(gain_pct(c(mae_dummy_val, mean(mae_clin$forest),  mean(mae_rf40_clin)),  mae_dummy_val), "%"),
  MAE_Combine   = round(c(mae_dummy_val, mean(mae_both$forest),  mean(mae_rf40_comb)),  2),
  Gain_Combine  = paste0(gain_pct(c(mae_dummy_val, mean(mae_both$forest),  mean(mae_rf40_comb)),  mae_dummy_val), "%"),
  row.names = NULL
)

par(mar = c(1, 1, 3, 1))
plot.new()
title(main = "MAE moyen et gain vs baseline dummy – RF complet vs RF top 40 (LOO)", cex.main = 1.1)
tbl      <- as.matrix(recap_rf40)
col_hdrs <- colnames(recap_rf40)
n_col    <- length(col_hdrs)
n_row    <- nrow(recap_rf40)
x_pos    <- seq(0.02, 0.98, length.out = n_col)
y_header <- 0.88
y_rows   <- seq(0.72, 0.08, length.out = n_row)
for (j in seq_along(col_hdrs)) {
  text(x_pos[j], y_header, col_hdrs[j], font = 2, cex = 0.85, adj = 0.5)
}
abline(h = 0.82, col = "gray40", lwd = 1.2)
for (i in seq_len(n_row)) {
  if (i %% 2 == 0) rect(0, y_rows[i] - 0.045, 1, y_rows[i] + 0.045,
                        col = "#f0f0f0", border = NA)
  for (j in seq_len(n_col)) {
    txt <- tbl[i, j]
    clr <- "black"
    if (grepl("Gain", col_hdrs[j])) {
      val <- as.numeric(sub("%", "", txt))
      clr <- ifelse(is.na(val), "black", ifelse(val > 0, "darkgreen", "red"))
    }
    text(x_pos[j], y_rows[i], txt, cex = 0.82, col = clr, adj = 0.5)
  }
}
abline(h = 0.78, col = "gray70", lwd = 0.8)
dev.off()

# PDF : boxplots accélérométriques uniquement (LOO2, RF complet vs RF top40, config accel)
pdf(out("9b_accel_boxplots.pdf"), width = 7, height = 6)
boxplot(list("RF complet\n(accel)" = mae_accel$forest,
             "RF top 40\n(accel)"  = mae_rf40_accel),
        col  = c("steelblue", "steelblue3"),
        main = "MAE – Features accélérométriques : RF complet vs RF top 40 (LOO)",
        ylab = "MAE", las = 1)
abline(h = mae_dummy_val, lty = 2, col = "red", lwd = 2)
legend("topright", legend = "Baseline dummy", lty = 2, col = "red", lwd = 2, bty = "n")
dev.off()




# ============================================================
# LOO 3 : TOP 40 FEATURES — TOUS LES MODELES
# ============================================================
# Complète le LOO 2 (RF seul) en évaluant tous les modèles sur le top 40 combiné
top_features <- head(imp_df$variable, 40)
top_features <- intersect(top_features, colnames(feat_both_clust))
feat_top40   <- feat_both_clust[, top_features]

mae_top40 <- run_loo(feat_top40, label, "Top 40 features", k_knn = 5)

print("Comparaison Random Forest :")
print(paste("Toutes features :", round(mean(mae_both$forest), 2)))
print(paste("Top 40 features :", round(mean(mae_top40$forest), 2)))


# ============================================================
# REDUCTION DIMENSIONNELLE PAR ACP
# ============================================================
print("Réduction par ACP")

ncp_max  <- min(65, ncol(feat_both_clust))
# LIMITE (fuite faible) : ACP ajustée sur tous les patients avant le LOO (non supervisée).
pca_full <- PCA(feat_both_clust, graph = FALSE, ncp = ncp_max)

composantes_a_tester <- c(5, 10, 20, 30, 40, 50)
mae_pca_results      <- data.frame()

for (ncp_val in composantes_a_tester) {
  if (ncp_val > ncp_max) next
  feat_pca <- as.data.frame(pca_full$ind$coord[, 1:ncp_val])
  mae_pca  <- run_loo(feat_pca, label, paste0("ACP ", ncp_val, " composantes"), k_knn = 5)
  mae_pca_results <- rbind(mae_pca_results,
                           data.frame(ncp      = ncp_val,
                                      mae_rf   = round(mean(mae_pca$forest), 3),
                                      mae_svm  = round(mean(mae_pca$svm), 3),
                                      mae_tree = round(mean(mae_pca$tree), 3)))
  print(paste("ACP", ncp_val, "composantes - MAE RF :", round(mean(mae_pca$forest), 2)))
}

print("Résultats réduction par ACP :")
print(mae_pca_results)

pdf(out("17_reduction_pca_resultats.pdf"), width = 8, height = 6)
print(ggplot(mae_pca_results, aes(x = ncp)) +
  geom_line(aes(y = mae_rf,   color = "Random Forest"), lwd = 1.2) +
  geom_line(aes(y = mae_svm,  color = "SVM"),           lwd = 1.2) +
  geom_line(aes(y = mae_tree, color = "Arbre"),         lwd = 1.2) +
  geom_point(aes(y = mae_rf,   color = "Random Forest"), size = 3) +
  geom_point(aes(y = mae_svm,  color = "SVM"),           size = 3) +
  geom_point(aes(y = mae_tree, color = "Arbre"),         size = 3) +
  geom_hline(yintercept = mean(mae_both$dummy), lty = 2, color = "gray40") +
  labs(title = "MAE selon nombre de composantes ACP",
       x = "Nombre de composantes", y = "MAE", color = "Modèle") +
  theme_minimal())
dev.off()


preds_rf <- numeric(n_patients)
for (i in 1:n_patients) {
  tr <- data.frame(feat_both_clust[-i, ], score_tampa = label[-i])
  mo <- randomForest(score_tampa ~ ., data = tr,
                     ntree = best_rf_params$ntree, mtry = best_rf_params$mtry)
  preds_rf[i] <- as.numeric(predict(mo, feat_both_clust[i, , drop = FALSE]))
}

erreurs <- abs(label - preds_rf)

acp_coords$erreur <- erreurs

p_acp_erreur <- ggplot(acp_coords, aes(x = Dim1, y = Dim2, color = erreur)) +
  geom_point(size = 3) +
  scale_color_gradient(low = "green", high = "red", name = "Erreur de prédiction") +
  labs(title = "ACP - Individus colorés par erreur de prédiction",
       x = "Dim 1", y = "Dim 2") +
  theme_minimal()

pdf(out("18_acp_individus_erreur.pdf"), width = 8, height = 6)
print(p_acp_erreur)
dev.off()

df_erreurs <- data.frame(
  sid         = df_merged$sid,
  score_tampa = label,
  prediction  = preds_rf,
  erreur      = erreurs
)
df_erreurs <- df_erreurs[order(-df_erreurs$erreur), ]

print("Top 10 patients les plus mal prédits :")
print(head(df_erreurs, 10))

seuil_erreur_elevee <- 10

pdf(out("19_predictions_vs_reel.pdf"), width = 8, height = 6)
plot(label, preds_rf, pch = 19,
     col  = ifelse(erreurs > seuil_erreur_elevee, "red", "steelblue"),
     xlab = "Score TAMPA réel", ylab = "Score TAMPA prédit",
     main = paste("Prédictions vs Réel (MAE =", round(mean(erreurs), 2), ")"))
abline(0, 1, lty = 2, col = "gray50", lwd = 2)
text(label[erreurs > seuil_erreur_elevee],
     preds_rf[erreurs > seuil_erreur_elevee],
     labels = df_merged$sid[erreurs > seuil_erreur_elevee],
     pos = 4, cex = 0.7, col = "red")
legend("topleft", legend = c("Bien prédit", "Erreur élevée"),
       col = c("steelblue", "red"), pch = 19)
dev.off()


mal_predits      <- erreurs > seuil_erreur_elevee
statut_prediction <- ifelse(mal_predits, "Mal prédit", "Bien prédit")
sid_mal_predits   <- df_merged$sid[mal_predits]

acp_coords_mal           <- as.data.frame(pca_result$ind$coord[, 1:2])
colnames(acp_coords_mal) <- c("Dim1", "Dim2")
acp_coords_mal$cluster_kmeans    <- as.factor(km3$cluster)
acp_coords_mal$cluster_hclust    <- as.factor(cutree(hclust3, k = 3))
acp_coords_mal$mal_predit        <- mal_predits
acp_coords_mal$sid               <- df_merged$sid
acp_coords_mal$statut_prediction <- statut_prediction

top_mal_predits <- acp_coords_mal[mal_predits, ]

p_kmeans_mal <- ggplot(acp_coords_mal, aes(x = Dim1, y = Dim2)) +
  geom_point(aes(color = cluster_kmeans, shape = statut_prediction), size = 3) +
  geom_point(data = top_mal_predits, aes(x = Dim1, y = Dim2),
             color = "red", size = 5, shape = 1, stroke = 1.5) +
  geom_text(data = top_mal_predits, aes(label = sid),
            color = "red", size = 3, nudge_y = 0.3, fontface = "bold") +
  scale_shape_manual(values = c("Bien prédit" = 16, "Mal prédit" = 17)) +
  labs(title = "K-means K=3 - Patients mal prédits mis en évidence",
       x = "Dim 1", y = "Dim 2",
       color = "Cluster", shape = "Prédiction") +
  theme_minimal()

pdf(out("20_clustering_kmeans_mal_predits.pdf"), width = 9, height = 7)
print(p_kmeans_mal)
dev.off()

p_hclust_mal <- ggplot(acp_coords_mal, aes(x = Dim1, y = Dim2)) +
  geom_point(aes(color = cluster_hclust, shape = statut_prediction), size = 3) +
  geom_point(data = top_mal_predits, aes(x = Dim1, y = Dim2),
             color = "red", size = 5, shape = 1, stroke = 1.5) +
  geom_text(data = top_mal_predits, aes(label = sid),
            color = "red", size = 3, nudge_y = 0.3, fontface = "bold") +
  scale_shape_manual(values = c("Bien prédit" = 16, "Mal prédit" = 17)) +
  labs(title = "Clustering hiérarchique K=3 - Patients mal prédits mis en évidence",
       x = "Dim 1", y = "Dim 2",
       color = "Cluster", shape = "Prédiction") +
  theme_minimal()

pdf(out("21_clustering_hclust_mal_predits.pdf"), width = 9, height = 7)
print(p_hclust_mal)
dev.off()

print(paste("Patients mal prédits (erreur >", seuil_erreur_elevee, ") :", sum(mal_predits)))
print("Répartition dans les clusters K-means :")
print(table(km3$cluster[mal_predits]))
print("Répartition dans les clusters hiérarchiques :")
print(table(cutree(hclust3, k = 3)[mal_predits]))


# Analyse descriptive des erreurs (pas de prédiction) : les scores de douleur et
# Oswestry peuvent être inclus ici sans risque de fuite
vars_cliniques <- c("age", "poids", "taille", "imc",
                    "evanuit", "evados", "evajamb", "score_oswestry")

df_wilcox <- df_merged[, c("sid", vars_cliniques)]
df_wilcox$mal_predit <- mal_predits

wilcox_results <- data.frame()
for (col_name in vars_cliniques) {
  vals_mal  <- df_wilcox[df_wilcox$mal_predit == TRUE,  col_name]
  vals_bien <- df_wilcox[df_wilcox$mal_predit == FALSE, col_name]
  test <- wilcox.test(vals_mal, vals_bien, exact = FALSE)
  wilcox_results <- rbind(wilcox_results, data.frame(
    variable     = col_name,
    mediane_mal  = round(median(vals_mal,  na.rm = TRUE), 2),
    mediane_bien = round(median(vals_bien, na.rm = TRUE), 2),
    W            = round(test$statistic, 2),
    p_value      = round(test$p.value, 4),
    significatif = ifelse(test$p.value < 0.05, "***", ifelse(test$p.value < 0.1, ".", ""))
  ))
}

print("Tests de Wilcoxon - Mal prédits vs Bien prédits :")
print(wilcox_results[order(wilcox_results$p_value), ])

if ("sexe" %in% colnames(df_merged)) {
  print("Test Fisher - Sexe :")
  print(fisher.test(table(df_merged$sexe, mal_predits)))
}

pdf(out("22_wilcoxon_clinique_mal_predits.pdf"), width = 12, height = 10)
par(mfrow = c(2, 4))
for (col_name in vars_cliniques) {
  vals_mal  <- df_wilcox[df_wilcox$mal_predit == TRUE,  col_name]
  vals_bien <- df_wilcox[df_wilcox$mal_predit == FALSE, col_name]
  p_val     <- wilcox_results$p_value[wilcox_results$variable == col_name]
  sig_label <- wilcox_results$significatif[wilcox_results$variable == col_name]
  boxplot(list("Bien prédit" = vals_bien, "Mal prédit" = vals_mal),
          col  = c("steelblue", "tomato"),
          main = paste0(col_name, "\np = ", p_val, " ", sig_label),
          ylab = col_name, las = 1)
}
dev.off()


# Arbre de décision pour caractériser les patients mal prédits
# On inclut toutes les variables cliniques disponibles y compris EVA et Oswestry
# car c'est de l'analyse descriptive, pas de la prédiction : pas de risque de leakage
mal_predit_binaire <- ifelse(erreurs > seuil_erreur_elevee, 1, 0)

print(paste("Patients mal prédits :", sum(mal_predit_binaire)))
print(paste("Patients bien prédits :", sum(mal_predit_binaire == 0)))

df_arbre_erreur <- data.frame(
  age            = df_merged$age,
  sexe           = df_merged$sexe,
  poids          = df_merged$poids,
  taille         = df_merged$taille,
  imc            = df_merged$imc,
  evanuit        = df_merged$evanuit,
  evados         = df_merged$evados,
  score_oswestry = df_merged$score_oswestry,
  nonwear_pct    = df_merged$nonwear_percentage,
  score_tampa    = label,
  mal_predit     = as.factor(mal_predit_binaire)
)

arbre_erreur <- rpart(mal_predit ~ age + sexe + poids + taille + imc +
                        evanuit + evados + score_oswestry + nonwear_pct + score_tampa,
                      data    = df_arbre_erreur,
                      method  = "class",
                      control = rpart.control(cp = 0, maxdepth = 3, minsplit = 2))

print("Structure de l'arbre sur les patients mal prédits :")
print(arbre_erreur$frame[, c("var", "n", "dev")])

pdf(out("23_arbre_patients_mal_predits.pdf"), width = 10, height = 8)
prp(arbre_erreur,
    extra         = 104,
    main          = "Arbre - Caractérisation des patients mal prédits",
    box.palette   = list("Gy", "Rd"),
    fallen.leaves = TRUE,
    varlen        = 0,
    faclen        = 0)
dev.off()

imp_arbre_erreur <- arbre_erreur$variable.importance
if (length(imp_arbre_erreur) > 0) {
  imp_arbre_erreur <- sort(imp_arbre_erreur, decreasing = TRUE)
  print("Variables les plus importantes pour distinguer les mal prédits :")
  print(round(imp_arbre_erreur, 2))
  
  pdf(out("24_importance_arbre_mal_predits.pdf"), width = 8, height = 5)
  barplot(imp_arbre_erreur,
          las  = 2,
          col  = "tomato",
          main = "Importance des variables — arbre patients mal prédits",
          ylab = "Importance")
  dev.off()
}


df_clin_tree    <- data.frame(feat_clin_raw, score_tampa = label)
tree_interprete <- rpart(score_tampa ~ ., data = df_clin_tree, method = "anova",
                         control = rpart.control(cp       = best_tree_params$cp,
                                                 minsplit = best_tree_params$minsplit))

print(paste("Arbre interprétable - noeuds :", nrow(tree_interprete$frame)))

pdf(out("25_arbre_decision_interpretable.pdf"), width = 10, height = 8)
prp(tree_interprete, extra = 1,
    main = "Arbre de décision interprétable (features cliniques brutes)",
    box.palette = "Blues", fallen.leaves = TRUE)
dev.off()


print(paste("Patients analysés         :", n_patients))
print(paste("Features accel            :", ncol(feat_accel)))
print(paste("Features clinique         :", ncol(feat_clin)))
print(paste("Features combiné+clusters :", ncol(feat_both_clust)))
print(paste("K optimal KNN             :", k_optimal))
print("MAE Random Forest (LOO) :")
print(paste("  Accel seul      :", round(mean(mae_accel$forest), 2)))
print(paste("  Clinique seul   :", round(mean(mae_clin$forest), 2)))
print(paste("  Combiné         :", round(mean(mae_both$forest), 2)))
print(paste("  Top 40 features :", round(mean(mae_top40$forest), 2)))
print(paste("  Baseline dummy  :", round(mean(mae_both$dummy), 2)))
print("Calibration SVM :") ;   print(best_svm_params)
print("Calibration Arbre :") ; print(best_tree_params)
print("Calibration RF :") ;    print(best_rf_params)
print("Réduction ACP :") ;     print(mae_pca_results)
