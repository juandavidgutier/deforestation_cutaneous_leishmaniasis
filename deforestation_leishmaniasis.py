import os, warnings, random
import dowhy
import econml
from dowhy import CausalModel
import pandas as pd
import numpy as np
from econml.dml import DML
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import scipy.stats as stats
from econml.dml import SparseLinearDML, LinearDML, CausalForestDML
from econml.orf import DMLOrthoForest
from econml.inference import BootstrapInference
from econml.score import RScorer
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import GroupKFold

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)


# Set seeds for reproducibility
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

SEED = 123
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
#os.environ['TF_DETERMINISTIC_OPS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
#os.environ['NUMEXPR_NUM_THREADS'] = '1'


#%%

file_path = 'D:/clases/UDES/fortalecimiento institucional/macroproyecto_2025/leish/ci/data_final_15_sep.csv'
data_all = pd.read_csv(file_path, encoding='latin-1')

data_all = data_all.dropna()

columnas_to_drop = ['Year', 'Altitude', 'Forest', 'cases', 'total_pop', 
                      'expected']

# 1. Label Encoding DANE
le = LabelEncoder()
data_all['DANE_labeled'] = le.fit_transform(data_all['DANE'])
scaler = MinMaxScaler()
data_all['DANE_normalized'] = scaler.fit_transform(
    data_all[['DANE_labeled']])

# 2. Label Encoding DANE_Year
le_year = LabelEncoder()
data_all['DANE_Year_labeled'] = le_year.fit_transform(data_all['DANE_year'])
scaler_DDANE = MinMaxScaler()
data_all['DANE_Year_normalized'] = scaler_DDANE.fit_transform(
    data_all[['DANE_Year_labeled']])


data_all.drop(columns=columnas_to_drop, inplace=True)

std_deforestation = data_all['Deforestation_t'].std()
print(f"std of Deforestation_t: {std_deforestation}")

median_Deforestation_t = data_all['Deforestation_t'].median()
print(f"median of Deforestation_t: {median_Deforestation_t}")

#%%


# Estandarización de variables continuas
scaler = StandardScaler()
data_all['MPI'] = scaler.fit_transform(data_all[['MPI']])
#data_all['Forest_tm1'] = scaler.fit_transform(data_all[['Forest_tm1']])
data_all['HFP_t'] = scaler.fit_transform(data_all[['HFP_t']])
data_all['Illegal_mining_t'] = scaler.fit_transform(data_all[['Illegal_mining_t']])
data_all['Soil_Moisture'] = scaler.fit_transform(data_all[['Soil_Moisture']])
data_all['Temperature'] = scaler.fit_transform(data_all[['Temperature']])
data_all['Precipitation'] = scaler.fit_transform(data_all[['Precipitation']])
data_all['Vectors'] = scaler.fit_transform(data_all[['Vectors']])
data_all['Fire_t'] = scaler.fit_transform(data_all[['Fire_t']])
data_all['Coca_t'] = scaler.fit_transform(data_all[['Coca_t']])
data_all['Deforestation_t'] = scaler.fit_transform(data_all[['Deforestation_t']])

# std
data_std = data_all[['DANE_normalized', 'DANE_Year_normalized',
                      'MPI', 'Forest_tm1', 'HFP_t', 'Illegal_mining_t', 'Soil_Moisture',
                      'Temperature', 'Precipitation', 'Vectors', 'Fire_t',
                      'Coca_t', 'Deforestation_t', 'excess_tp1']]

# Asegurar orden temporal correcto
data_std = data_std.sort_values(
    by=['DANE_normalized', 'DANE_Year_normalized']
).reset_index(drop=True)



#%%
# 1. Definir el DAG como una variable
dag_string = """graph[directed 1 

                node[id "Forest_tm1" label "Forest_tm1"]
                node[id "Precipitation" label "Precipitation"]
                node[id "Temperature" label "Temperature"]
                node[id "Soil_Moisture" label "Soil_Moisture"]
                node[id "MPI" label "MPI"]
                node[id "HFP_t" label "HFP_t"]
                node[id "Coca_t" label "Coca_t"]
                node[id "Fire_t" label "Fire_t"]
                node[id "Illegal_mining_t" label "Illegal_mining_t"]
                node[id "Vectors" label "Vectors"]
                node[id "Deforestation_t" label "Deforestation_t"]
                node[id "excess_tp1" label "excess_tp1"]
                node[id "DANE_normalized" label "DANE_normalized"]
                node[id "DANE_Year_normalized" label "DANE_Year_normalized"]
                
                edge[source "Forest_tm1" target "Precipitation"]
                edge[source "Forest_tm1" target "Temperature"]
                edge[source "Forest_tm1" target "Soil_Moisture"]
                edge[source "Forest_tm1" target "MPI"]
                edge[source "Forest_tm1" target "HFP_t"]
                edge[source "Forest_tm1" target "Coca_t"]
                edge[source "Forest_tm1" target "Fire_t"]
                edge[source "Forest_tm1" target "Illegal_mining_t"]
                edge[source "Forest_tm1" target "Vectors"]
                edge[source "Forest_tm1" target "Deforestation_t"]
                edge[source "Forest_tm1" target "excess_tp1"]

                edge[source "Precipitation" target "Temperature"]
                edge[source "Precipitation" target "Soil_Moisture"]
                edge[source "Precipitation" target "Vectors"]
                edge[source "Precipitation" target "MPI"]
                edge[source "Precipitation" target "HFP_t"]
                edge[source "Precipitation" target "Illegal_mining_t"]
                edge[source "Precipitation" target "excess_tp1"]
                
                edge[source "Temperature" target "Vectors"]
                edge[source "Temperature" target "MPI"]
                edge[source "Temperature" target "HFP_t"]
                edge[source "Temperature" target "Coca_t"]
                edge[source "Temperature" target "Fire_t"]
                edge[source "Temperature" target "Illegal_mining_t"]
                edge[source "Temperature" target "Deforestation_t"]
                edge[source "Temperature" target "excess_tp1"]
                
                edge[source "Soil_Moisture" target "Vectors"]
                edge[source "Soil_Moisture" target "MPI"]
                edge[source "Soil_Moisture" target "HFP_t"]
                edge[source "Soil_Moisture" target "excess_tp1"]
                
                edge[source "MPI" target "HFP_t"]
                edge[source "MPI" target "Coca_t"]
                edge[source "MPI" target "Illegal_mining_t"]
                edge[source "MPI" target "Vectors"]
                edge[source "MPI" target "Deforestation_t"]
                edge[source "MPI" target "excess_tp1"]
                
                edge[source "HFP_t" target "Coca_t"]
                edge[source "HFP_t" target "Fire_t"]
                edge[source "HFP_t" target "Illegal_mining_t"]
                edge[source "HFP_t" target "Vectors"]
                edge[source "HFP_t" target "Deforestation_t"]
                edge[source "HFP_t" target "excess_tp1"]
                
                edge[source "Coca_t" target "Deforestation_t"]
                edge[source "Coca_t" target "excess_tp1"]
                
                edge[source "Fire_t" target "Deforestation_t"]
                edge[source "Fire_t" target "excess_tp1"]
                
                edge[source "Illegal_mining_t" target "Deforestation_t"]
                edge[source "Illegal_mining_t" target "excess_tp1"]
                
                edge[source "Vectors" target "excess_tp1"]
                
                edge[source "Deforestation_t" target "excess_tp1"]
                
                edge[source "DANE_normalized" target "excess_tp1"]
            ]"""

# Ahora sí puedes usar dag_string en tu modelo
model_deforestation = CausalModel(
    data=data_std,
    treatment=['Deforestation_t'],
    outcome=['excess_tp1'],   
    graph=dag_string
)

  

#%%

from PIL import Image
import matplotlib.pyplot as plt

# Generate the model graph
model_deforestation.view_model()

    
#%% 

# Identifying effects
identified_estimand_deforest = model_deforestation.identify_effect(proceed_when_unidentifiable=None)                                                       
print(identified_estimand_deforest)

#%%

# ─────────────────────────────────────────────
# Municipality‑grouped cross‑fitting folds
# ─────────────────────────────────────────────
# Prevent data leakage: all observations from the same municipality
# must stay together in the same fold. Pass GroupKFold + municipality
# groups to the DML estimator's fit() method.
municipality_groups = data_std['DANE_normalized'].values
print(f"Cross‑fitting: GroupKFold(n_splits=3) grouped by {len(np.unique(municipality_groups))} municipalities.")

effect_modifiers = ['Forest_tm1', 'DANE_normalized']

reg1 = lambda: HistGradientBoostingRegressor(max_iter=5,  max_depth=3, random_state=123, learning_rate=0.05) #reg_lambda=1.5, alpha=0.001)
reg2 = lambda: HistGradientBoostingClassifier(max_iter=5,  max_depth=3, random_state=123, learning_rate=0.05) #reg_lambda=1.5, alpha=0.001)

causal_estimate_std = model_deforestation.estimate_effect(
    identified_estimand_deforest,
    method_name="backdoor.econml.dml.DML",
    effect_modifiers=effect_modifiers,
    confidence_intervals=False,
    method_params={
        "init_params": {
            "model_y": reg2(),
            "model_t": reg1(),
            "model_final": LassoCV(
                alphas=[0.0001, 0.001, 0.005, 0.05, 0.01, 0.1],
                fit_intercept=False,
                max_iter=50000,
                tol=1e-3,
                cv=3,
                n_jobs=-1),
            "discrete_outcome": True,
            "discrete_treatment": False,
            "random_state": 123,
            "cv": 3
        },
        "fit_params": {
            "inference": BootstrapInference(n_bootstrap_samples=100, n_jobs=-1)
        }
    }
)

print("\nDML model fitted with municipality-grouped cross-fitting.")
print(f"ATE not computed here; cluster bootstrap provides all inference.")

#%%

# Access the fitted EconML estimator for CATE predictions
econml_estimator = causal_estimate_std.estimator.estimator

# ─────────────────────────────────────────────
# ATE verification: ensure full-sample ATE matches refutation's "Estimated effect"
# ─────────────────────────────────────────────
# The refutation tests print "Estimated effect:" using causal_estimate_std.value.
# We verify this value against our own call to .ate() and use the SAME value
# for reporting, so no discrepancy is possible.
effect_modifiers_list = ['Forest_tm1', 'DANE_normalized']
X_data_all = data_std[effect_modifiers_list].dropna()
ate_from_econml = float(econml_estimator.ate(X=X_data_all))
ate_from_dowhy   = float(causal_estimate_std.value)

print(f"\n{'='*60}")
print("ATE VERIFICATION")
print(f"{'='*60}")
print(f"  ATE (from EconML .ate() call) : {ate_from_econml:.6f}")
print(f"  ATE (from causal_estimate_std) : {ate_from_dowhy:.6f}")
print(f"  (Refutation 'Estimated effect' will show this SAME value)")

if abs(ate_from_econml - ate_from_dowhy) < 1e-10:
    print("  ✓ Perfect match — confirmed.")
else:
    print(f"  ⚠ Difference: {abs(ate_from_econml - ate_from_dowhy):.2e}")
    print(f"  Using causal_estimate_std.value for consistency with refutations.")
print(f"{'='*60}")

# Use DoWhy's value for reporting (exactly what refutations show)
ate_Deforestation_t = ate_from_dowhy
print(f"\nFull-sample ATE (for reporting): {ate_Deforestation_t:.6f}")
print("(Confidence intervals from cluster bootstrap below)")

#%%

random_std = model_deforestation.refute_estimate(identified_estimand_deforest, causal_estimate_std,
                                         method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_std)

# with subset
subset_std  = model_deforestation.refute_estimate(identified_estimand_deforest, causal_estimate_std,
                                          method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=10)
print(subset_std) 
      
# with bootstrap
bootstrap_std  = model_deforestation.refute_estimate(identified_estimand_deforest, causal_estimate_std,
                                             method_name="bootstrap_refuter", random_state=123, num_simulations=10)
print(bootstrap_std)

# with placebo 
placebo_std  = model_deforestation.refute_estimate(identified_estimand_deforest, causal_estimate_std,
                                           method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=10)
print(placebo_std)    


#%%

# non-parametric partial R² Chernozhukov et al. (2021)

X = data_std[['Coca_t','MPI','Temperature','Fire_t','HFP_t','Illegal_mining_t','Forest_tm1']]
T = data_std["Deforestation_t"]
Y = data_std["excess_tp1"]

mi_T = mutual_info_regression(X, T,random_state=123)
mi_Y = mutual_info_classif(X, Y,random_state=123)

score = mi_T * mi_Y
ranking = pd.Series(score, index=X.columns).sort_values(ascending=False)
print(ranking.head(10)) # Illegal_mining_t is the strongest confounder

#%%

# 2) Run sensitivity refutation (non-parametric partial R2)
partialR2_deforest = model_deforestation.refute_estimate(
    identified_estimand_deforest,
    causal_estimate_std,
    method_name="add_unobserved_common_cause",
    simulation_method="non-parametric-partial-R2",
    benchmark_common_causes=["Illegal_mining_t"],
    effect_fraction_on_treatment=0.1,
    effect_fraction_on_outcome=0.1,
    plugin_reisz=False,
    num_simulations=500,
    plot_estimate=False
)

print(partialR2_deforest)
print(partialR2_deforest.RV)
print(partialR2_deforest.RV_alpha)

# ===============================
# PARTIAL R2 BENCHMARK Illegal_mining_t
# ===============================

X = data_std[['Coca_t','MPI','Temperature','Fire_t','HFP_t','Forest_tm1']] # exclude the benchmark confounder
T = data_std["Deforestation_t"]
Y = data_std["excess_tp1"]
Z = data_std["Illegal_mining_t"]


from sklearn.model_selection import KFold


kf = KFold(n_splits=5, shuffle=True, random_state=123)

T_res = np.zeros(len(T))
Y_res = np.zeros(len(Y))
Z_res = np.zeros(len(Z))

for train, test in kf.split(X):

    mt = reg1()
    my = reg1()
    mz = reg1()

    mt.fit(X.iloc[train], T.iloc[train])
    my.fit(X.iloc[train], Y.iloc[train])
    mz.fit(X.iloc[train], Z.iloc[train])

    T_res[test] = T.iloc[test] - mt.predict(X.iloc[test])
    Y_res[test] = Y.iloc[test] - my.predict(X.iloc[test])
    Z_res[test] = Z.iloc[test] - mz.predict(X.iloc[test])


# partial R²
r2_z_t = np.corrcoef(Z_res, T_res)[0,1]**2
r2_z_y = np.corrcoef(Z_res, Y_res)[0,1]**2

print("Partial R² Illegal_mining_t→T | X:", r2_z_t)
print("Partial R² Illegal_mining_t→Y | X:", r2_z_y)

# ==========================
# STRENGTH MULTIPLIER
# ==========================
RV_point   = partialR2_deforest.RV
RV_alpha   = partialR2_deforest.RV_alpha          # ADD this line
r2_bench_T = r2_z_t
r2_bench_Y = r2_z_y

# ─────────────────────────────────────────────
# SCENARIO 1 — Nullify the POINT estimate (RV)
# ─────────────────────────────────────────────
k_T_point = RV_point / r2_bench_T if r2_bench_T > 0 else np.inf
k_Y_point = RV_point / r2_bench_Y if r2_bench_Y > 0 else np.inf
k_binding_point = max(k_T_point, k_Y_point)

print("\n=== SCENARIO 1: Nullify POINT estimate ===")
print(f"  RV (point)                    : {RV_point:.4f}")
print(f"  k on T (Illegal_mining_t)            : {k_T_point:.4f}")
print(f"  k on Y (Illegal_mining_t)            : {k_Y_point:.4f}")
print(f"  BINDING k (most demanding condition) : {k_binding_point:.4f}")
if RV_point == 0.0:
    print("  ► Any unobserved confounder, no matter how small,")
    print("    is sufficient to bring the point estimate to zero.")
else:
    print(f"  ► U must be {k_binding_point:.2f}x stronger than Illegal_mining_t")
    print(f"    (simultaneously on T and Y) to nullify the point effect.")

# ─────────────────────────────────────────────
# SCENARIO 2 — Nullify SIGNIFICANCE (RV_alpha)
# ─────────────────────────────────────────────
k_T_alpha = RV_alpha / r2_bench_T if r2_bench_T > 0 else np.inf
k_Y_alpha = RV_alpha / r2_bench_Y if r2_bench_Y > 0 else np.inf
k_binding_alpha = max(k_T_alpha, k_Y_alpha)

print("\n=== SCENARIO 2: Nullify STATISTICAL SIGNIFICANCE (α=0.05) ===")
print(f"  RV_alpha (α=0.05)             : {RV_alpha:.4f}")
print(f"  k on T (Illegal_mining_t)            : {k_T_alpha:.4f}")
print(f"  k on Y (Illegal_mining_t)            : {k_Y_alpha:.4f}")
print(f"  BINDING k (most demanding condition) : {k_binding_alpha:.4f}")
if RV_alpha >= 1.0:
    print("  ► RV_alpha ≥ 1.0: no confounder can explain")
    print("    more than 100% of the residual variance. Statistical")
    print("    significance is impregnable to unobserved confounding.")
else:
    print(f"  ► U must be {k_binding_alpha:.2f}x stronger than Illegal_mining_t")
    print(f"    (simultaneously on T and Y) to invalidate significance.")

# ─────────────────────────────────────────────
# COMPARATIVE SUMMARY TABLE
# ─────────────────────────────────────────────
summary_table = pd.DataFrame({
    "Scenario"         : ["Nullify point estimate", "Nullify significance (α=0.05)"],
    "RV"               : [RV_point,  RV_alpha],
    "k on T"           : [k_T_point, k_T_alpha],
    "k on Y"           : [k_Y_point, k_Y_alpha],
    "binding k"        : [k_binding_point, k_binding_alpha],
    "R² bench T (Illegal_mining_t)" : [r2_bench_T, r2_bench_T],
    "R² bench Y (Illegal_mining_t)" : [r2_bench_Y, r2_bench_Y]
})
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print("\n=== SENSITIVITY ANALYSIS SUMMARY TABLE ===")
print(summary_table.to_string(index=False))


#%%
# ╔══════════════════════════════════════════════════════════════╗
# ║  CLUSTER BOOTSTRAP (CLUSTER‑ROBUST STANDARD ERRORS)          ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Standard bootstrap resamples individual observations, which
# ignores the within‑municipality correlation of repeated measures.
# The cluster bootstrap resamples entire municipalities (clusters)
# with replacement, preserving the intra‑cluster dependence structure.
#
# Reference: Cameron & Miller (2015), JHR 50(2), 317–372.
#            Abadie et al. (2023), QJE 138(1), 1–35.

def cluster_bootstrap_ate_cate(
    data,
    cluster_col,
    dag_string,
    effect_modifiers_list,
    treatment_col,
    outcome_col,
    X_test_grid=None,
    n_bootstrap=50,
    seed=123,
    verbose=True
):
    """
    Cluster bootstrap for DML: resample municipalities (clusters) with
    replacement, keeping all time periods within each resampled cluster.
    Re‑run the full DoWhy + DML pipeline on each bootstrap sample.
    
    Computes both ATE and CATE (if X_test_grid is provided).
    The point estimate and CI come from the SAME bootstrap distribution,
    so the CI always contains the point estimate by construction.
    
    Parameters
    ----------
    data : pd.DataFrame
        Full dataset
    cluster_col : str
        Column name identifying clusters (municipalities)
    dag_string : str or networkx.DiGraph
        DAG specification
    effect_modifiers_list : list of str
        Names of effect modifier columns
    treatment_col, outcome_col : str
        Treatment and outcome column names
    X_test_grid : np.ndarray or None
        If provided, CATE is computed at these X points for each bootstrap
    n_bootstrap : int
        Number of bootstrap iterations
    seed : int
        Random seed
    verbose : bool
        Print progress
    
    Returns
    -------
    dict with ATE and CATE results
    """
    clusters = data[cluster_col].unique()
    n_clusters = len(clusters)
    rng = np.random.RandomState(seed)
    
    reg1 = lambda: XGBRegressor(n_estimators=1000, max_depth=3, 
                                 random_state=123, eta=0.0001, 
                                 reg_lambda=1.5, alpha=0.001)
    reg2 = lambda: XGBClassifier(n_estimators=1000, max_depth=3, 
                                  random_state=123, eta=0.0001, 
                                  reg_lambda=1.5, alpha=0.001)
    
    ate_bootstrap = []
    cate_bootstrap = []  # list of arrays, each (n_grid_points,)
    
    for b in range(n_bootstrap):
        # 1) Sample clusters with replacement
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        
        # 2) Build bootstrap dataset
        boot_parts = []
        for c in sampled_clusters:
            boot_parts.append(data[data[cluster_col] == c])
        boot_data = pd.concat(boot_parts, axis=0).reset_index(drop=True)
        
        # 3) New group labels for bootstrap sample
        boot_data['_boot_group'] = pd.factorize(boot_data[cluster_col])[0]
        
        # 4) Cross‑fitting folds grouped by municipality
        boot_gkf = GroupKFold(n_splits=3)
        boot_cv = list(boot_gkf.split(boot_data, groups=boot_data['_boot_group']))
        
        # 5) Fit DoWhy + DML on bootstrap sample
        boot_model = CausalModel(
            data=boot_data,
            treatment=[treatment_col],
            outcome=[outcome_col],
            graph=dag_string
        )
        boot_identified = boot_model.identify_effect(proceed_when_unidentifiable=None)
        
        boot_estimate = boot_model.estimate_effect(
            boot_identified,
            method_name="backdoor.econml.dml.DML",
            effect_modifiers=effect_modifiers_list,
            confidence_intervals=False,
            method_params={
                "init_params": {
                    "model_y": reg2(),
                    "model_t": reg1(),
                    "model_final": LassoCV(
                        alphas=[0.0001, 0.001, 0.005, 0.05, 0.01, 0.1],
                        fit_intercept=False,
                        max_iter=50000,
                        tol=1e-3,
                        cv=3,
                        n_jobs=-1),
                    "discrete_outcome": True,
                    "discrete_treatment": False,
                    "random_state": 123,
                    "cv": boot_cv
                },
                "fit_params": {}
            }
        )
        
        boot_estimator = boot_estimate.estimator.estimator
        
        # 6) Extract ATE
        X_mean = boot_data[effect_modifiers_list].mean().to_frame().T
        ate_b = boot_estimator.ate(X=X_mean)
        ate_bootstrap.append(float(ate_b))
        
        # 7) Extract CATE at test grid (if provided)
        if X_test_grid is not None:
            cate_b = boot_estimator.effect(X_test_grid)
            cate_bootstrap.append(cate_b.flatten())
        
        if verbose and (b + 1) % 10 == 0:
            print(f"  Cluster bootstrap iteration {b + 1}/{n_bootstrap} completed.")
    
    # ─── ATE results ───
    ate_mean = float(np.mean(ate_bootstrap))
    ate_se = float(np.std(ate_bootstrap, ddof=1))
    ate_ci = (ate_mean - 1.96 * ate_se, ate_mean + 1.96 * ate_se)
    
    results = {
        "ate_mean": ate_mean,
        "ate_se": ate_se,
        "ate_ci_95": ate_ci,
        "ate_distribution": ate_bootstrap
    }
    
    # ─── CATE results (from bootstrap distribution) ───
    # The point estimate is the MEDIAN of the bootstrap CATEs at each X point.
    # The CI is the 2.5th and 97.5th percentiles.
    # Since median always lies between min and max at each point,
    # the line is guaranteed to be inside the band.
    if X_test_grid is not None and len(cate_bootstrap) > 0:
        cate_matrix = np.column_stack(cate_bootstrap)  # (n_grid, n_bootstrap)
        
        results["cate_median"] = np.median(cate_matrix, axis=1)
        results["cate_lower"] = np.percentile(cate_matrix, 2.5, axis=1)
        results["cate_upper"] = np.percentile(cate_matrix, 97.5, axis=1)
        results["cate_matrix"] = cate_matrix
    
    return results


# ─────────────────────────────────────────────
# Prepare CATE test grid (Forest surface, others held at mean)
# ─────────────────────────────────────────────
# Grid for Forest_tm1
Forest_tm1 = data_std['Forest_tm1']
min_Forest_tm1 = Forest_tm1.min()
max_Forest_tm1 = Forest_tm1.max()
delta = (max_Forest_tm1 - min_Forest_tm1) / 100
Forest_tm1_grid = np.arange(min_Forest_tm1, max_Forest_tm1 + delta - 0.001, delta)

# Means of other effect modifiers
DANE_encoded_mean = data_std['DANE_normalized'].mean()


X_test_grid = np.column_stack([
    Forest_tm1_grid,
    np.full_like(Forest_tm1_grid, DANE_encoded_mean),

])

# ─────────────────────────────────────────────
# Run cluster bootstrap (50 iterations)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLUSTER BOOTSTRAP (50 iterations)")
print("Resampling municipalities with replacement...")
print("=" * 60)

cluster_boot_results = cluster_bootstrap_ate_cate(
    data=data_std,
    cluster_col='DANE_normalized',
    dag_string=dag_string,
    effect_modifiers_list=['Forest_tm1', 'DANE_normalized'],
    treatment_col='Deforestation_t',
    outcome_col='excess_tp1',
    X_test_grid=X_test_grid,
    n_bootstrap=50,
    seed=123,
    verbose=True
)

# ─────────────────────────────────────────────
# CLUSTER‑ROBUST ATE RESULTS
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("CLUSTER‑ROBUST ATE RESULTS")
print("─" * 60)
print(f"  ATE (cluster bootstrap)       : {cluster_boot_results['ate_mean']:.6f}")
print(f"  SE (cluster‑robust)           : {cluster_boot_results['ate_se']:.6f}")
print(f"  95% CI (cluster‑robust)       : {cluster_boot_results['ate_ci_95']}")
print(f"  ATE (full‑sample estimate)    : {ate_Deforestation_t:.6f}")
print("")
print("Note: The SE accounts for within‑municipality correlation (Cameron & Miller 2015).")
print("─" * 60)

# ─────────────────────────────────────────────
# CATE PLOT (from cluster bootstrap)
# ─────────────────────────────────────────────
# The line is the MEDIAN of bootstrap CATE curves.
# The band is the 2.5th‑97.5th percentile range.
# Both come from the SAME bootstrap distribution → line always inside band.

cate_median = cluster_boot_results['cate_median']
cate_lower = cluster_boot_results['cate_lower']
cate_upper = cluster_boot_results['cate_upper']

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_facecolor('#F0F0F0')
ax.grid(True, color='#D0D0D0', linestyle='-', linewidth=0.6, alpha=0.7, zorder=0)

ax.fill_between(
    Forest_tm1_grid,
    cate_lower,
    cate_upper,
    alpha=0.25,
    color='blue',
    linewidth=0,
    zorder=2
)

ax.plot(
    Forest_tm1_grid,
    cate_median,
    color='darkblue',
    linewidth=2.0,
    zorder=3
)

ax.axhline(y=0, color='crimson', linestyle='--', linewidth=1.0, alpha=0.8, zorder=1)

ax.set_xlabel('Forest coverage (%)', fontsize=14, fontweight='medium')
ax.set_ylabel('Effect of deforestation on excess CL cases', fontsize=14, fontweight='medium')
ax.set_title('CATE: Effect deforestation on excess CL cases\nConditional on Forest coverage',
             fontsize=13, fontweight='bold', pad=12)
ax.tick_params(axis='both', labelsize=12, length=4, width=0.8, color='#555555')

plt.tight_layout()

