import random
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from econml.dr import SparseLinearDRLearner, ForestDRLearner, LinearDRLearner
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy.stats import expon
import scipy.stats as stats
from scipy.interpolate import interp1d
import statsmodels.api as sm
from dowhy import CausalModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBClassifier
from dowhy.causal_estimator import CausalEstimate
from econml.dr import DRLearner
from sklearn.linear_model import LassoCV
from econml.dml import DML, SparseLinearDML
from PIL import Image

# Set seeds for reproducibility
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')

#%% 
# --- Data Loading and Cleaning ---
data_all = pd.read_csv("D:/data.csv", encoding='latin-1')
data_all = data_all.dropna()

# Drop unnecessary columns for the causal model
columns_to_drop = ['Year', 'Altitude', 'Forest', 'cases', 'total_pop', 
                    'expected', 'sir']
data_all.drop(columns=columns_to_drop, inplace=True)

#%% 
# --- Feature Engineering and Normalization ---

# 1. Label Encoding and Normalization for DANE (Spatial identifier)
le = LabelEncoder()
data_all['DANE_labeled'] = le.fit_transform(data_all['DANE'])
scaler = MinMaxScaler()
data_all['DANE_normalized'] = scaler.fit_transform(data_all[['DANE_labeled']])

# 2. Label Encoding and Normalization for DANE_Year (Spatio-temporal identifier)
le_year = LabelEncoder()
data_all['DANE_Year_labeled'] = le_year.fit_transform(data_all['DANE_year'])
scaler_DDANE = MinMaxScaler()
data_all['DANE_Year_normalized'] = scaler_DDANE.fit_transform(data_all[['DANE_Year_labeled']])

# Statistical summary of the treatment variable
std_deforestation = data_all['Deforestation_t'].std()
print(f"Standard deviation of deforestation: {std_deforestation}")
median_deforestation = data_all['Deforestation_t'].median()
print(f"Median of deforestation: {median_deforestation}")

# Standardization of continuous variables (Confounders and Covariates)
scaler = StandardScaler()
continuous_vars = ['MPI', 'HFP_t', 'Illegal_mining_t', 'Soil_Moisture', 
                   'Temperature', 'Precipitation', 'Vectors', 'Fire_t', 
                   'Coca_t', 'Deforestation_t']

for var in continuous_vars:
    data_all[var] = scaler.fit_transform(data_all[[var]])

# Standardized Dataset selection
data_std = data_all[['DANE_normalized', 'DANE_Year_normalized',
                      'MPI', 'Forest_tm1', 'HFP_t', 'Illegal_mining_t', 'Soil_Moisture',
                      'Temperature', 'Precipitation', 'Vectors', 'Fire_t',
                      'Coca_t', 'Deforestation_t', 'Excess_cases_tp1']]

#%% 
# --- Causal Model Specification (DAG) ---

model_deforestation = CausalModel(
    data=data_std,
    treatment=['Deforestation_t'],
    outcome=['Excess_cases_tp1'],
    graph="""graph[directed 1 
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
                node[id "Excess_cases_tp1" label "Excess_cases_tp1"]
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
                edge[source "Forest_tm1" target "Excess_cases_tp1"]

                edge[source "Precipitation" target "Temperature"]
                edge[source "Precipitation" target "Soil_Moisture"]
                edge[source "Precipitation" target "Vectors"]
                edge[source "Precipitation" target "MPI"]
                edge[source "Precipitation" target "HFP_t"]
                edge[source "Precipitation" target "Illegal_mining_t"]
                edge[source "Precipitation" target "Excess_cases_tp1"]
                
                edge[source "Temperature" target "Vectors"]
                edge[source "Temperature" target "MPI"]
                edge[source "Temperature" target "HFP_t"]
                edge[source "Temperature" target "Coca_t"]
                edge[source "Temperature" target "Fire_t"]
                edge[source "Temperature" target "Illegal_mining_t"]
                edge[source "Temperature" target "Deforestation_t"]
                edge[source "Temperature" target "Excess_cases_tp1"]
                
                edge[source "Soil_Moisture" target "Vectors"]
                edge[source "Soil_Moisture" target "MPI"]
                edge[source "Soil_Moisture" target "HFP_t"]
                edge[source "Soil_Moisture" target "Excess_cases_tp1"]
                
                edge[source "MPI" target "HFP_t"]
                edge[source "MPI" target "Coca_t"]
                edge[source "MPI" target "Illegal_mining_t"]
                edge[source "MPI" target "Vectors"]
                edge[source "MPI" target "Deforestation_t"]
                edge[source "MPI" target "Excess_cases_tp1"]
                
                edge[source "HFP_t" target "Coca_t"]
                edge[source "HFP_t" target "Fire_t"]
                edge[source "HFP_t" target "Illegal_mining_t"]
                edge[source "HFP_t" target "Vectors"]
                edge[source "HFP_t" target "Deforestation_t"]
                edge[source "HFP_t" target "Excess_cases_tp1"]
                
                edge[source "Coca_t" target "Deforestation_t"]
                edge[source "Coca_t" target "Excess_cases_tp1"]
                
                edge[source "Fire_t" target "Deforestation_t"]
                edge[source "Fire_t" target "Excess_cases_tp1"]
                
                edge[source "Illegal_mining_t" target "Deforestation_t"]
                edge[source "Illegal_mining_t" target "Excess_cases_tp1"]
                
                edge[source "Vectors" target "Excess_cases_tp1"]
                edge[source "Deforestation_t" target "Excess_cases_tp1"]
                edge[source "DANE_normalized" target "Excess_cases_tp1"]
                edge[source "DANE_Year_normalized" target "Excess_cases_tp1"]
            ]"""
)

# Visualizing the DAG
model_deforestation.view_model()
img = Image.open("causal_model.png")
plt.figure(figsize=(16, 12))
plt.imshow(img)
plt.axis('off')
plt.title("DAG: Deforestation and Leishmaniasis (Acyclic Model)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#%% 
# --- Identification of Causal Estimand ---
print("\n" + "=" * 70)
print("IDENTIFYING CAUSAL ESTIMAND")
print("=" * 70)

identified_estimand_deforest = model_deforestation.identify_effect(
    proceed_when_unidentifiable=True
)
print(identified_estimand_deforest)

#%% 
# --- Estimation using Sparse Linear DML (Double Machine Learning) ---
reg1 = lambda: XGBRegressor(n_estimators=3400, max_depth=35, random_state=123, 
                            eta=0.0001, reg_lambda=1.5, alpha=0.001)

causal_estimate_std = model_deforestation.estimate_effect(
    identified_estimand_deforest,
    method_name="backdoor.econml.dml.SparseLinearDML",
    effect_modifiers=['Forest_tm1', 'DANE_normalized', 'DANE_Year_normalized'],
    confidence_intervals=True,
    method_params={
        "init_params": {
            "model_y": reg1(),
            "model_t": reg1(),                                                                                                                                                                                                                                 
            "discrete_outcome": True,
            "discrete_treatment": False,
            "max_iter": 50000,
            "tol": 1e-4,
            "alpha": 'auto',
            "cv": 5,
            "random_state": 123
        },
    }
)

#%% 
# --- Average Treatment Effect (ATE) and Confidence Intervals ---
econml_estimator = causal_estimate_std.estimator.estimator
effect_modifiers = ['Forest_tm1', 'DANE_normalized', 'DANE_Year_normalized']  

X_data_deforest = data_std[effect_modifiers].dropna()  
ate_deforest = econml_estimator.ate(X=X_data_deforest)
ate_ci_deforest = econml_estimator.ate_interval(X=X_data_deforest, alpha=0.05)

print(f"  ATE: {ate_deforest}")
print(f"  95% CI of ATE: {ate_ci_deforest}")

#%% 
# --- Conditional Average Treatment Effect (CATE) Visualization ---

# 1. Grid setup for Forest_tm1
forest = data_std['Forest_tm1']
min_forest = forest.min()
max_forest = forest.max()
delta = (max_forest - min_forest) / 100
forest_grid = np.arange(min_forest, max_forest + delta - 0.001, delta)

# 2. Covariate means for ceteris paribus prediction
DANE_encoded_mean = data_std['DANE_normalized'].mean()
DANE_Year_encoded_mean = data_std['DANE_Year_normalized'].mean()

X_test_grid = np.column_stack([
    forest_grid,
    np.full_like(forest_grid, DANE_encoded_mean),
    np.full_like(forest_grid, DANE_Year_encoded_mean)   
])

# 3. Predict effect and confidence intervals
treatment_effect = econml_estimator.effect(X_test_grid)
hte_lower2_cons, hte_upper2_cons = econml_estimator.effect_interval(X_test_grid, alpha=0.05)

plot_data = pd.DataFrame({
    'forest': forest_grid,
    'treatment_effect': treatment_effect.flatten(),
    'hte_lower2_cons': hte_lower2_cons.flatten(),
    'hte_upper2_cons': hte_upper2_cons.flatten()
})

# 4. Matplotlib Visualization (Academic Style with Light Gray Background)
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})
bg_color = '#f5f5f5'

fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
ax.set_facecolor(bg_color)

# Plotting CATE and CI
ax.plot(plot_data['forest'], plot_data['treatment_effect'], color='blue', lw=2, label='CATE', zorder=3)
ax.fill_between(plot_data['forest'], plot_data['hte_lower2_cons'], plot_data['hte_upper2_cons'], 
                color='blue', alpha=0.2, label='95% CI', zorder=2)

ax.axhline(y=0, color='red', linestyle='--', lw=1.2, label='Zero Effect', zorder=1)
ax.set_xlabel('Forest coverage (%)')
ax.set_ylabel('Effect of Deforestation on Excess CL Cases')
ax.grid(True, color='white', linestyle='-', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
plt.tight_layout()
plt.show()

#%% 
# --- Refutation Tests (Robustness Checks) ---

# Random Common Cause
random_std = model_deforestation.refute_estimate(identified_estimand_deforest, causal_estimate_std,
                                         method_name="random_common_cause", random_state=123, num_simulations=50)
print(random_std)

# Data Subset Refuter
subset_std = model_deforestation.refute_estimate(identified_estimand_deforest, causal_estimate_std,
                                          method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=50)
print(subset_std) 
      
# Bootstrap Refuter
bootstrap_std = model_deforestation.refute_estimate(identified_estimand_deforest, causal_estimate_std,
                                             method_name="bootstrap_refuter", random_state=123, num_simulations=50)
print(bootstrap_std)

# Placebo Treatment Refuter
placebo_std = model_deforestation.refute_estimate(identified_estimand_deforest, causal_estimate_std,
                                           method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=50)
print(placebo_std)    

#%% 
# --- Sensitivity Analysis (E-Value Calculation) ---

# STEP 1: Define exposure levels for comparison (Standardized units)
deforest_std_min = -0.68
deforest_std_max = 13.77

# STEP 2: Predict causal effects at exposure extremes
effect_at_min = econml_estimator.effect(X=X_data_deforest, T0=0, T1=deforest_std_min)
effect_at_max = econml_estimator.effect(X=X_data_deforest, T0=0, T1=deforest_std_max)

mean_effect_at_min = np.mean(effect_at_min)
mean_effect_at_max = np.mean(effect_at_max)
effect_diff = mean_effect_at_max - mean_effect_at_min

# STEP 3: Calculate baseline prevalence
baseline_risk = data_std['Excess_cases_tp1'].mean()
print(f"\nBaseline Prevalence of Excess_cases_tp1: {baseline_risk:.6f}")

# STEP 4: Estimate risks at each exposure level
risk_at_min = np.clip(baseline_risk + mean_effect_at_min, 0.0001, 0.9999)
risk_at_max = np.clip(baseline_risk + mean_effect_at_max, 0.0001, 0.9999)

# STEP 5: Calculate Risk Ratio (RR) in log scale
log_RR = np.log(risk_at_max) - np.log(risk_at_min)
RR_point_estimate = np.exp(log_RR)

# STEP 6: Delta Method for Standard Error of log(RR)
effect_interval_min = econml_estimator.effect_interval(X=X_data_deforest, T0=0, T1=deforest_std_min, alpha=0.05)
effect_interval_max = econml_estimator.effect_interval(X=X_data_deforest, T0=0, T1=deforest_std_max, alpha=0.05)

se_effect_min = (np.mean(effect_interval_min[1]) - np.mean(effect_interval_min[0])) / (2 * 1.96)
se_effect_max = (np.mean(effect_interval_max[1]) - np.mean(effect_interval_max[0])) / (2 * 1.96)

# Variance of log(RR) using Delta Method: Var[log(P)] ≈ [1/P]^2 * Var[P]
var_log_RR = ((1 / risk_at_max)**2 * se_effect_max**2) + ((-1 / risk_at_min)**2 * se_effect_min**2)
se_log_RR = np.sqrt(var_log_RR)

# STEP 7: Confidence Intervals for RR
log_RR_lower = log_RR - 1.96 * se_log_RR
log_RR_upper = log_RR + 1.96 * se_log_RR
RR_lower_CI = np.exp(log_RR_lower)
RR_upper_CI = np.exp(log_RR_upper)

# STEP 8: Final Summary for E-Value Analysis
param_evalue_deforest_final = pd.DataFrame({
    'Analysis': ['deforestation_effect'],
    'RR_point_estimate': [RR_point_estimate],
    'RR_lower_CI': [RR_lower_CI],
    'RR_upper_CI': [RR_upper_CI],
    'log_RR': [log_RR],
    'se_log_RR': [se_log_RR],
    'baseline_risk': [baseline_risk]
})

print("\n" + "="*80)
print("FINAL SUMMARY FOR E-VALUE ANALYSIS")
print("="*80)
print(param_evalue_deforest_final[['Analysis', 'RR_point_estimate', 'RR_lower_CI', 'RR_upper_CI']].to_string(index=False))

# Export results
output_path = "D:/param_evalue_deforest.csv"
param_evalue_deforest_final.to_csv(output_path, index=False)