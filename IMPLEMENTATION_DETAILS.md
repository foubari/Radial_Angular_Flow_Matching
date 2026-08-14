# Annexe : Détails d'implémentation

Ce document fournit l'ensemble des détails nécessaires à la reproduction complète des expériences présentées dans ce travail.

---

## 1. Architecture du réseau neuronal

### 1.1 Structure

Toutes les méthodes basées sur le Flow Matching (Gaussian FM, source-only, RAFM) ainsi que la baseline MSGM utilisent la **même architecture** pour assurer une comparaison équitable.

| Composant | Spécification |
|---|---|
| Type | Perceptron multi-couches (MLP) |
| Couche d'entrée | Linear(d + 1, 128) + Swish |
| Couches cachées | 2 × [Linear(128, 128) + Swish] |
| Couche de sortie | Linear(128, d), sans activation |
| Profondeur totale | 4 couches linéaires (3 avec activation + 1 sortie) |
| Largeur | 128 unités (constante) |
| Activation | Swish : x · σ(x), où σ est la sigmoïde |
| Normalisation | Aucune (ni BatchNorm, ni LayerNorm) |
| Dropout | Aucun |
| Connexions résiduelles | Aucune |
| Biais | Activé sur toutes les couches linéaires |
| Initialisation des poids | Défaut PyTorch : U[-√k, √k] avec k = 1/fan_in |

### 1.2 Entrée et sortie

- **Entrée** : concaténation de x ∈ ℝ^d et du scalaire t ∈ [0, 1], formant un vecteur de dimension d + 1.
- **Sortie** : champ de vitesse v_θ(x, t) ∈ ℝ^d, de même dimension que les données.

Le temps t est directement concaténé au vecteur d'entrée (pas d'embedding appris, pas de sinusoïdal).

### 1.3 Nombre de paramètres

| Dimension d | Paramètres |
|---|---|
| d = 2 | ≈ 33 920 |
| d = 16 | ≈ 35 712 |
| d = 32 | ≈ 37 760 |
| d = 256 | ≈ 66 304 |

---

## 2. Formulation du Conditional Flow Matching

### 2.1 Objectif

La perte CFM (Conditional Flow Matching) est :

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - u_t(x_0, x_1, t) \|_2^2 \right]$$

où :
- t ~ Uniform[0, 1]
- x_1 est tiré du jeu d'entraînement (données cibles)
- x_0 est tiré de la distribution source
- x_t = path(x_0, x_1, t) est l'interpolation au temps t
- u_t est le champ de vitesse conditionnel analytique

La perte est calculée comme la MSE par dimension, sommée sur les dimensions puis moyennée sur le batch. Aucune pondération temporelle n'est appliquée.

### 2.2 Chemin euclidien (Gaussian FM, source-only)

Interpolation linéaire :

$$x_t = (1 - t) \, x_0 + t \, x_1$$

Champ de vitesse conditionnel (constant en t) :

$$u_t(x_0, x_1, t) = x_1 - x_0$$

### 2.3 Chemin géodésique sphérique (RAFM)

Les points source et cible sont sur la même sphère de rayon R = ‖x_1‖. Soit u_0 = x_0/R et u_1 = x_1/R les vecteurs unitaires correspondants, et θ = arccos(⟨u_0, u_1⟩) l'angle entre eux.

**Interpolation par SLERP :**

$$x_t = R \left[ \frac{\sin((1-t)\theta)}{\sin\theta} \, u_0 + \frac{\sin(t\theta)}{\sin\theta} \, u_1 \right]$$

**Champ de vitesse conditionnel :**

$$u_t = R \cdot \frac{\theta}{\sin\theta} \left[ -\cos((1-t)\theta) \, u_0 + \cos(t\theta) \, u_1 \right]$$

Ce qui correspond au logarithme riemannien : u_t = (1/(1-t)) · Log_{x_t}^R(x_1).

**Cas dégénérés** :
- θ < 10⁻⁶ (quasi-identiques) : repli sur interpolation linéaire
- θ > π - 10⁻³ (quasi-antipodaux) : perturbation de u_0 par un vecteur perpendiculaire aléatoire (amplitude 10⁻³)
- Toutes les divisions utilisent un clamp minimal de 10⁻¹² pour la stabilité numérique

### 2.4 Couplage source-cible (méthodes radiales)

Pour les sources radiales (oracle ou empirique), le couplage est :

1. Extraire R = ‖x_1‖ (rayon du point cible)
2. Tirer u_0 ~ Uniform(S^{d-1}) (direction aléatoire sur la sphère unité, via normalisation gaussienne)
3. Définir x_0 = R · u_0

Ce couplage garantit ‖x_0‖ = ‖x_1‖ : la source et la cible partagent la même sphère.

### 2.5 Projection tangente (échantillonnage sphérique)

Lors de l'intégration ODE pour l'échantillonnage avec le chemin géodésique, la composante radiale du champ prédit est supprimée à chaque pas :

$$v_{\text{proj}} = v - \frac{\langle x, v \rangle}{\|x\|^2} \, x$$

Cela force ‖x_t‖ = R pour tout t, empêchant la dérive de norme. Pour ‖x‖ < r_min (r_min = 10⁻³), la projection n'est pas appliquée (sphère dégénérée près de l'origine).

---

## 3. Distributions sources

### 3.1 Source gaussienne

$$x_0 \sim \mathcal{N}(0, I_d)$$

Source standard sans paramètres. Utilisée par Gaussian FM (baseline).

### 3.2 Source radiale oracle

$$x_0 = R \cdot u_0, \quad R \sim p_R^{\text{exact}}, \quad u_0 \sim \text{Uniform}(S^{d-1})$$

Le rayon R est tiré de la distribution radiale analytique exacte de la cible :
- **Student-t corrélée** : R = ‖A z‖ où z ~ Student-t(df, d) i.i.d., A est la matrice de mélange
- **Gaussien anisotrope** : R = ‖A z‖ où z ~ N(0, I_d)
- **Toy 2D** : R = |Student-t(df)| (valeur absolue)

Disponible uniquement pour les jeux de données synthétiques (CDF analytique connue).

### 3.3 Source radiale empirique (eCDF)

$$x_0 = R \cdot u_0, \quad R \sim \hat{F}_R^{-1}(U), \quad U \sim \text{Uniform}[0,1]$$

- Estimation à partir des **données d'entraînement uniquement** (pas de fuite de données)
- Calcul des rayons : r_i = ‖x_i^{\text{train}}‖ pour tout i
- Échantillonnage par inversion de la CDF empirique : R = quantile(r_train, U) avec interpolation linéaire
- Les rayons négatifs (impossibles) sont clampés à 0

**Variante log-rayon** (expérience 2) : estimation sur log(r + 10⁻⁶), puis exponentiation après échantillonnage.

---

## 4. Entraînement

### 4.1 Hyperparamètres

| Paramètre | Valeur |
|---|---|
| Optimiseur | Adam |
| Taux d'apprentissage | 0.001 |
| β₁, β₂ | 0.9, 0.999 (défaut PyTorch) |
| ε | 10⁻⁸ (défaut PyTorch) |
| Weight decay | 0 |
| Taille de batch | 4096 |
| Nombre de pas | 10 000 |
| Scheduler | Aucun (LR constant) |
| Warmup | Aucun |
| Gradient clipping | Aucun |
| EMA | Non utilisé |

### 4.2 Procédure

1. Les données d'entraînement sont pré-chargées entièrement sur GPU
2. À chaque pas, un mini-batch est tiré par échantillonnage aléatoire avec remplacement via `torch.randint`
3. Pas de DataLoader PyTorch : indexation directe dans le tenseur GPU
4. `torch.compile` activé sur Linux/CUDA (désactivé sur Windows)
5. Sauvegarde de checkpoint tous les 5 000 pas

### 4.3 Convergence

Des tests de convergence sur Student-t d=16 montrent que radial_W1 reste identique à 10k, 50k et 100k pas (≈ 0.25-0.28), car cette métrique est entièrement déterminée par la source (la projection tangente fixe la norme). Le sliced_W1 montre une légère amélioration entre 10k et 50k pas.

---

## 5. Échantillonnage (inférence)

### 5.1 Intégration ODE

L'échantillonnage résout l'ODE déterministe :

$$\frac{dx}{dt} = v_\theta(x, t), \quad t \in [0, 1]$$

avec x_0 tiré de la distribution source.

### 5.2 Solveurs disponibles

**Euler (ordre 1)** :
$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$$
NFE = N pas.

**Heun (ordre 2, prédicteur-correcteur)** :
$$\tilde{x} = x_t + \Delta t \cdot v_\theta(x_t, t)$$
$$x_{t+\Delta t} = x_t + \frac{\Delta t}{2} \left[ v_\theta(x_t, t) + v_\theta(\tilde{x}, t + \Delta t) \right]$$
NFE = 2N pas.

**RK4 (ordre 4, défaut)** :
$$k_1 = v_\theta(x_t, t)$$
$$k_2 = v_\theta(x_t + \frac{\Delta t}{2} k_1, t + \frac{\Delta t}{2})$$
$$k_3 = v_\theta(x_t + \frac{\Delta t}{2} k_2, t + \frac{\Delta t}{2})$$
$$k_4 = v_\theta(x_t + \Delta t \cdot k_3, t + \Delta t)$$
$$x_{t+\Delta t} = x_t + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$
NFE = 4N pas.

### 5.3 Configuration par défaut

| Paramètre | Valeur |
|---|---|
| Solveur | RK4 |
| Nombre de pas N | 128 |
| NFE effective | 512 (= 4 × 128) |
| Projection tangente | Oui (chemin sphérique), Non (euclidien) |
| Nombre d'échantillons générés | 10 000 |

---

## 6. Jeux de données

### 6.1 Student-t corrélée

$$X = z \, A^\top, \quad z_i \sim \text{Student-t}(\nu), \; i.i.d.$$

| Paramètre | Valeur |
|---|---|
| Dimensions testées | d = 16, d = 32 |
| Degrés de liberté ν | 3.0 |
| Nombre d'échantillons | 50 000 |
| Corrélation | Oui (matrice A aléatoire) |
| Matrice de mélange A | A ∈ ℝ^{d×d}, A_{ij} ~ N(0,1), seed = 42 |
| Propriétés | Queues lourdes, polynomiales ∝ ‖x‖^{-(ν+d)} |

La matrice A est fixée à la construction et identique pour toutes les runs (reproductibilité via `matrix_seed = 42`).

### 6.2 Gaussien anisotrope (contrôle)

$$X = z \, A^\top, \quad z \sim \mathcal{N}(0, I_d)$$

| Paramètre | Valeur |
|---|---|
| Dimensions testées | d = 16, d = 32 |
| Nombre d'échantillons | 50 000 |
| Corrélation | Oui (même matrice A que Student-t) |

Jeu de données de contrôle : le Flow Matching gaussien standard devrait bien fonctionner. Permet de vérifier que RAFM ne dégrade pas les performances sur données à queues légères.

### 6.3 Toy 2D (radial-angulaire)

$$X = r \cdot [\cos\theta, \sin\theta]^\top$$

| Composante | Distribution |
|---|---|
| Rayon r | r = |Student-t(ν = 3)| × scale |
| Angle θ | Mélange de n_modes distributions von Mises |
| Nombre de modes | 4 |
| Concentration κ | 5.0 |
| Centres des modes | Uniformément espacés sur [0, 2π) |
| Scale | 1.0 |
| Nombre d'échantillons | 50 000 |

L'attribution de chaque point à un mode est uniforme. L'angle est échantillonné par approximation gaussienne de la von Mises : θ ~ N(μ_k, 1/κ).

### 6.4 PIV (données réelles)

**Source** : « Non-time-resolved PIV dataset of flow over a circular cylinder at Re = 3900 » (DOI: 10.57745/DHJXM6).

#### 6.4.1 Fichiers utilisés

Tous les fichiers du zip `dataverse_files.zip` dont le nom commence par `Serie_` et se termine par `.txt` sont retenus. Aucun filtrage par index : le seul critère d'exclusion est la présence de valeurs NaN dans Vx ou Vy, ou une erreur de parsing (nombre de points ≠ NX × NY = 545 × 740 = 403 300). Le nombre exact de snapshots retenus dépend du contenu du zip téléchargé et est affiché par le script (`Processed X frames, skipped Y`).

#### 6.4.2 Format des fichiers DaVis

Chaque fichier `.txt` contient un en-tête DaVis suivi de NX × NY = 403 300 lignes, chacune au format `x;y;Vx;Vy`. L'ordonnancement est row-major : y varie lentement (NY = 740 blocs), x varie rapidement (NX = 545 points par bloc). Les colonnes x et y (positions en mm) sont ignorées ; seuls Vx et Vy (en m/s) sont extraits et réorganisés en tableaux (NY, NX) = (740, 545).

#### 6.4.3 Calcul de la vorticité

La vorticité est calculée sur la **grille complète** (740 × 545) **avant** sous-échantillonnage :

$$\omega = \frac{\partial V_y}{\partial x} - \frac{\partial V_x}{\partial y}$$

Les dérivées sont calculées par `numpy.gradient` :
- `dVy/dx = numpy.gradient(Vy, axis=1)` — dérivée selon l'axe x (axe rapide, NX = 545)
- `dVx/dy = numpy.gradient(Vx, axis=0)` — dérivée selon l'axe y (axe lent, NY = 740)

**Pas spatiaux** : `numpy.gradient` est appelé **sans argument de pas spatial** (Δx, Δy). Par défaut, `numpy.gradient` utilise un pas unitaire (Δ = 1). Les valeurs de vorticité sont donc en unités de [m/s] / [pixel], pas en [1/s]. Cela n'affecte pas les résultats car la normalisation ultérieure (/2.5) absorbe l'échelle.

**Schéma de différences** : `numpy.gradient` utilise des différences centrales d'ordre 2 à l'intérieur du domaine, et des différences finies d'ordre 1 (avant/arrière) sur les bords.

#### 6.4.4 Sous-échantillonnage spatial

Le champ de vorticité ω ∈ ℝ^{740×545} est sous-échantillonné sur une grille (ny, nx) :

1. **Calcul des indices** :
   - `y_idx = numpy.linspace(0, 739, ny, dtype=int)` — ny indices uniformément espacés entre 0 et NY−1 inclus, arrondis par troncature vers zéro (cast `int`)
   - `x_idx = numpy.linspace(0, 544, nx, dtype=int)` — idem pour x

2. **Extraction** : `omega[numpy.ix_(y_idx, x_idx)]` — produit la sous-grille (ny, nx) par indexation croisée

3. **Aplatissement** : `.flatten()` en ordre C (row-major) : les indices x varient en premier au sein de chaque ligne y

**Grilles et dimensions résultantes** :

| Grille (ny × nx) | Dimension d | Indices y | Indices x |
|---|---|---|---|
| 8 × 4 | 32 | linspace(0, 739, 8).astype(int) | linspace(0, 544, 4).astype(int) |
| 8 × 8 | 64 | linspace(0, 739, 8).astype(int) | linspace(0, 544, 8).astype(int) |
| 16 × 16 | 256 | linspace(0, 739, 16).astype(int) | linspace(0, 544, 16).astype(int) |

#### 6.4.5 Normalisation et centrage

La normalisation est appliquée **sur l'ensemble du dataset** (tous les snapshots, avant tout split train/val/test) :

1. **Division par 2.5** : `data = data / 2.5` (constante reprise de la référence MSGM pour assurer la comparabilité)
2. **Centrage** : `data = data - data.mean(axis=0)` — soustraction de la moyenne empirique par dimension, calculée sur **tous les snapshots**

**Au chargement** (`piv.py`), un second centrage est appliqué après troncation dimensionnelle : `data = data - data.mean(dim=0)`. Ce second centrage opère aussi sur **l'ensemble du dataset avant split**. Pour les grilles natives (sans troncation), ce second centrage est quasi-nul car les données sont déjà centrées.

**Important** : les statistiques de normalisation (moyenne) sont calculées sur l'ensemble du dataset, pas uniquement sur le split d'entraînement. C'est un choix de design hérité de la baseline MSGM.

#### 6.4.6 Ordre exact du pipeline

```
1. Lecture des fichiers Serie_*.txt depuis le zip
2. Pour chaque snapshot :
   a. Parser Vx, Vy → tableaux (740, 545)
   b. Exclure si NaN dans Vx ou Vy
   c. Calculer ω = numpy.gradient(Vy, axis=1) - numpy.gradient(Vx, axis=0)
      sur la grille complète (740, 545), pas unitaire
   d. Sous-échantillonner : omega[ix_(y_idx, x_idx)].flatten()  → vecteur (d,)
3. Empiler tous les snapshots → tableau (N, d)
4. Normaliser : data /= 2.5
5. Centrer : data -= data.mean(axis=0)     [sur tout le dataset]
6. Sauvegarder en piv_d{dim}.pt (torch.float32)
7. Sauvegarder les versions tronquées piv_d{dim}_trunc{d'}.pt pour d' < d

--- Au chargement (piv.py) ---
8. Charger piv_d{dim}.pt
9. Tronquer aux d premières dimensions : data = data[:, :d]
10. Re-centrer : data -= data.mean(dim=0)   [sur tout le dataset]
11. Appliquer le split train/val/test (60/20/20, split_seed=0)
```

#### 6.4.7 Variantes PIV dans les expériences

| Variante | Grille native | Source du .pt | Dims utilisées | Exp |
|---|---|---|---|---|
| PIV d=256 | 16×16 | piv_d256.pt | 256 | Exp 1 |
| PIV d=64 | 8×8 | piv_d64.pt | 64 | Exp 1 |
| PIV d=32 | 8×4 | piv_d32.pt | 32 | Exp 1 |
| PIV d=16 | 8×4 | piv_d32.pt, tronqué | 16 premières | Exp 1 |

Les variantes d=16 et d=32 (depuis piv_d64 ou piv_d256) utilisent les versions tronquées `piv_d{dim}_trunc{d}.pt` ou la troncation au chargement.

#### 6.4.8 Format des fichiers intermédiaires

| Fichier | Shape | dtype | Contenu |
|---|---|---|---|
| piv_d32.pt | (N, 32) | float32 | Vorticité normalisée et centrée |
| piv_d64.pt | (N, 64) | float32 | Idem, grille 8×8 |
| piv_d256.pt | (N, 256) | float32 | Idem, grille 16×16 |
| piv_d64_trunc16.pt | (N, 16) | float32 | 16 premières colonnes de piv_d64 |
| piv_d256_trunc16.pt | (N, 16) | float32 | 16 premières colonnes de piv_d256 |
| piv_d256_trunc32.pt | (N, 32) | float32 | 32 premières colonnes de piv_d256 |

Les fichiers contiennent uniquement le tenseur de données X. La moyenne de centrage n'est pas stockée séparément. Les données ne sont pas mélangées (l'ordre des snapshots est celui du zip). Le split est appliqué au chargement.

#### 6.4.9 Commande de reconstruction

```bash
python -m rafm.data.prepare_piv \
    --zip dataverse_files.zip \
    --out_dir data/piv \
    --grids 8x4,8x8,16x16
```

**Note** : pas de source radiale oracle disponible pour ce jeu de données réel (CDF analytique inconnue). Seules les variantes empirique et gaussienne sont testées.

### 6.5 Partitionnement des données

| Paramètre | Valeur |
|---|---|
| Split | 60 % train / 20 % validation / 20 % test |
| Méthode | Permutation aléatoire des indices |
| Seed de partition | split_seed = 0 (indépendant du seed modèle) |
| Taille train (synthétique) | 30 000 |
| Taille validation | 10 000 |
| Taille test | 10 000 |

Les partitions sont disjointes et déterministes pour un split_seed donné.

### 6.6 Prétraitement des données synthétiques

**Aucun** : les données Student-t, gaussiennes et toy 2D sont utilisées directement sans centrage, normalisation ou augmentation.

### 6.7 Augmentation de données

**Aucune** augmentation n'est appliquée sur aucun jeu de données.

---

## 7. Baseline MSGM (Multiplicative Score-Generative Model)

### 7.1 Formulation SDE

L'EDS forward (bruit multiplicatif) en forme de Stratonovich est :

$$dY = G(Y) \circ dB_t$$

où G est un tenseur d×d×d à matrices antisymétriques.

**Construction de G** :
1. Générer d matrices aléatoires F_k ∈ ℝ^{d×d}, F_k ~ N(0, 1)
2. Antisymétriser : G[:,:,k] = (F_k − F_k^⊤) / 2
3. Normaliser : calculer L_G = (1/2) Σ_k G[:,:,k]² (tenseur de correction d'Itô), puis G ← √(−d / (2 tr(L_G))) · G

**En forme d'Itô** :
- Dérive : f(t, y) = L_G · β(t) · y
- Diffusion : g(t, y) = G · √(β(t)) · y (matrice d×d par échantillon)
- Divergence : div_Σ(t, y) = 2 L_G · β(t) · y

### 7.2 Schedule de bruit

$$\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min}) \cdot t$$

| Jeu de données | β_min | β_max |
|---|---|---|
| Student-t (défaut) | 0.1 | 20.0 |
| Student-t corrélée | 0.01 | 1.0 |
| PIV | 0.025 | 5.0 |

### 7.3 Échantillonnage du prior (latent)

MSGM utilise aussi une correction radiale via eCDF :

1. Calculer r_i = ‖x_i^{\text{train}}‖ pour tout i
2. Optionnel : transformer en log-rayon : r ← log(r + 10⁻⁶)
3. Estimer la KDE gaussienne (bandwidth = 0.1 × std(r))
4. Au sampling : R ~ quantile(r_train, U) (mode eCDF) ou R ~ KDE
5. Si log-rayon : R ← exp(R) − 10⁻⁶
6. Coupler avec une direction aléatoire : x_0 = R · u, u ~ Uniform(S^{d-1})

### 7.4 Perte d'entraînement

Sliced Score Matching (SSM) : estimation du score via l'EDS reverse plug-in avec estimation de trace de Hutchinson.

### 7.5 Hyperparamètres MSGM

| Paramètre | Valeur |
|---|---|
| Optimiseur | Adam, lr = 0.001 |
| Taille de batch | 4096 |
| Nombre de pas | 10 000 |
| Pas forward (coupling) | 16 (défaut), 128 (Student-t corrélée) |
| Architecture | MLP identique (128 hidden, 3 couches, Swish) |
| Checkpoint | Tous les 1 000 pas |

### 7.6 Échantillonnage MSGM

Solveur RK4 de Stratonovich (4ème ordre) avec bruit multiplicatif :

À chaque pas :
1. Tirer dW ~ N(0, √Δt · I)
2. Calculer 4 étapes (k₁ à k₄) combinant la dérive de Stratonovich et la diffusion appliquée à dW
3. Mettre à jour : x ← x + (k₁ + 2k₂ + 2k₃ + k₄) / 6

NFE par défaut : 128 pas. Correction de norme optionnelle : x_t ← x_t · (‖x_0‖ / ‖x_t‖).

---

## 8. Métriques d'évaluation

### 8.1 Métriques radiales

**Wasserstein-1 radiale (radial_W1)** :

$$W_1(\hat{F}_R^{\text{gen}}, \hat{F}_R^{\text{test}}) = \int_0^\infty |\hat{F}_R^{\text{gen}}(r) - \hat{F}_R^{\text{test}}(r)| \, dr$$

Calculé à partir des rayons ‖x‖ des échantillons générés et de test.

**Statistique de Kolmogorov-Smirnov (KS)** :

$$KS = \sup_r |\hat{F}_R^{\text{gen}}(r) - \hat{F}_R^{\text{test}}(r)|$$

**Erreurs de quantiles** aux niveaux 95 %, 99 %, 99.5 % :

$$\text{err}_q = \frac{|Q_{\text{gen}}(q) - Q_{\text{test}}(q)|}{|Q_{\text{test}}(q)| + 10^{-12}}$$

**Calibration d'excédance de queue** aux niveaux 95 %, 99 % :

$$\text{exc}_q = |P(R_{\text{gen}} > r_q) - P(R_{\text{test}} > r_q)|$$

### 8.2 Sliced Wasserstein-1 (espace complet)

$$SW_1(P, Q) = \frac{1}{L} \sum_{\ell=1}^{L} W_1(\langle \theta_\ell, \cdot \rangle_\# P, \langle \theta_\ell, \cdot \rangle_\# Q)$$

où θ_ℓ ~ Uniform(S^{d-1}) sont L = 500 directions de projection aléatoires. La W1 unidimensionnelle est calculée par tri et appariement des projections.

### 8.3 Métriques angulaires

Les échantillons sont partitionnés en n_bins = 4 bins de rayon (quantiles du test). Dans chaque bin, les vecteurs sont normalisés sur la sphère unité, puis la Sliced Wasserstein est calculée avec 200 projections.

Sortie : SW par bin + moyenne.

### 8.4 Maximum Mean Discrepancy (MMD)

$$\text{MMD}^2(X, Y) = \mathbb{E}[\kappa(X,X)] + \mathbb{E}[\kappa(Y,Y)] - 2\mathbb{E}[\kappa(X,Y)]$$

Noyau RBF gaussien avec bandwidth sélectionnée par heuristique de la médiane des distances pairwise.

### 8.5 Métriques de stabilité

| Métrique | Définition |
|---|---|
| NaN rate | Fraction d'échantillons contenant NaN |
| Exploding norm rate | Fraction avec ‖x‖ > 100 × médiane(‖x_test‖) |
| Invalid rate | nan_rate + (1 − nan_rate) × exploding_rate |

---

## 9. Protocole expérimental

### 9.1 Vue d'ensemble

| Expérience | Objectif | Jeux de données |
|---|---|---|
| Exp 1 : Benchmark principal | Comparer toutes les méthodes | Student-t d=16/32, Gaussien d=16/32, PIV d=16/32/64/256, Toy 2D |
| Exp 2 : Efficacité d'échantillon | Convergence vs taille d'entraînement | Student-t d=16 |
| Exp 3 : Temps de calcul | Temps par pas et par échantillonnage | Student-t d=16, d=32 |
| Exp 4 : Sensibilité au solveur | Qualité vs NFE et type de solveur | Student-t d=16 |

### 9.2 Méthodes testées (Exp 1)

| Méthode | Source | Chemin |
|---|---|---|
| Gaussian FM | N(0, I) | Euclidien |
| Source-only (oracle) | Radiale oracle | Euclidien |
| Source-only (empirique) | Radiale eCDF | Euclidien |
| RAFM (oracle) | Radiale oracle | Géodésique sphérique |
| RAFM (empirique) | Radiale eCDF | Géodésique sphérique |
| MSGM | Multiplicatif (eCDF) | — (SDE) |

Pour PIV : pas de variante oracle (CDF analytique inconnue).

### 9.3 Exp 2 : Efficacité d'échantillon

Tailles d'entraînement testées : n_train ∈ {500, 1 000, 5 000, 20 000, 50 000}.
Méthodes : Gaussian FM, source-only (oracle, empirique eCDF, empirique log), RAFM (oracle, empirique eCDF).

### 9.4 Exp 3 : Temps de calcul

Mesure du temps par pas d'entraînement (ms/step) et du temps d'échantillonnage à NFE ∈ {32, 64, 128, 256}. 3 runs de timing, 1 seed.

### 9.5 Exp 4 : Sensibilité au solveur

Solveurs : Euler, Heun, RK4.
NFE ∈ {4, 8, 16, 32, 64, 128, 256}.
Modèle pré-entraîné (RAFM empirique, Student-t d=16).

### 9.6 Ablation : projection tangente

Reprise des checkpoints RAFM entraînés, rééchantillonnage avec `project_tangent = False`. Testé sur : Toy 2D, Student-t d=16/d=32, Gaussien d=16, PIV d=16/d=64/d=256.

---

## 10. Reproductibilité

### 10.1 Gestion des seeds

| Seed | Rôle | Valeur par défaut |
|---|---|---|
| base_seed | Initialisation modèle, échantillonnage | 42 |
| split_seed | Partition train/val/test | 0 |
| matrix_seed | Génération de la matrice A | 42 |

Les 3 seeds sont indépendants. Pour chaque expérience, n_seeds = 3 runs sont effectuées avec les seeds {8925, 77395, 65457} (générés par `numpy.random.default_rng(42).integers(0, 100_000, size=3)`).

### 10.2 Dépendances logicielles

- Python 3.10+
- PyTorch (avec CUDA pour GPU)
- NumPy, SciPy
- scikit-learn (pour KernelDensity dans MSGM et source radiale empirique KDE)
- tqdm (progression)

### 10.3 Configuration matérielle

- `device: auto` sélectionne CUDA si disponible, sinon CPU
- `torch.compile` activé uniquement sur Linux avec CUDA
- Mémoire GPU : ≈ 39 MB (toutes les méthodes FM)

### 10.4 Paramètres de métriques

| Paramètre | Valeur |
|---|---|
| Projections Sliced Wasserstein | 500 |
| Projections angulaires | 200 |
| Bins angulaires | 4 |
| Facteur norme explosive | 100 |
| Échantillons générés pour évaluation | 10 000 |

---

## 11. Tableau récapitulatif de tous les hyperparamètres

| Catégorie | Paramètre | RAFM / FM | MSGM |
|---|---|---|---|
| **Architecture** | Hidden dim | 128 | 128 |
| | Couches | 3 | 3 |
| | Activation | Swish | Swish |
| | Time embedding | Concaténation | Concaténation |
| **Entraînement** | Optimiseur | Adam | Adam |
| | Learning rate | 10⁻³ | 10⁻³ |
| | Batch size | 4096 | 4096 |
| | Pas d'entraînement | 10 000 | 10 000 |
| | Weight decay | 0 | 0 |
| | Scheduler | Aucun | Aucun |
| | EMA | Non | Non |
| | Gradient clipping | Non | Non |
| **Échantillonnage** | Solveur | RK4 (ODE) | RK4 Stratonovich (SDE) |
| | Pas d'intégration | 128 | 128 |
| | Stochastique | Non | Oui |
| **Données** | Split | 60/20/20 | 60/20/20 |
| | n_samples (synth.) | 50 000 | 50 000 |
| | n_gen | 10 000 | 10 000 |
| **Seeds** | Modèle | 8925, 77395, 65457 | 8925, 77395, 65457 |
| | Partition | 0 | 0 |
| | Matrice A | 42 | 42 |
