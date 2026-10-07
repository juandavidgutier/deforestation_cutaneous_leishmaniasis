

# Importing required libraries
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
from econml.score import RScorer
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.linear_model import RidgeCV

# Compatibility with modern versions of numpy
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

# Set seeds for reproducibility
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Path to the file on Google Drive
file_path = 'D:/data.csv'  # Adjust the path if it is in a folder
data_all = pd.read_csv(file_path, encoding='latin-1')

data_all = data_all[data_all['Altitude'] <= 1700]

print(f"Dimensions after dropna: {data_all.shape}")

# Drop columns not needed for the causal analysis
columnas_to_drop = ['Altitude', 'Forest', 'cases', 'total_pop',
                     'expected', 'sir', 'Excess_cases']
data_all.drop(columns=columnas_to_drop, inplace=True, errors='ignore')

# 1. Label Encoding for DANE (municipality identifier)
le = LabelEncoder()
data_all['DANE_labeled'] = le.fit_transform(data_all['DANE'])
scaler = MinMaxScaler()
data_all['DANE_normalized'] = scaler.fit_transform(data_all[['DANE_labeled']])

# 2. Label Encoding for DANE_Year (time identifier)
le_year = LabelEncoder()
data_all['DANE_Year_labeled'] = le_year.fit_transform(data_all['DANE_year'])
scaler_DDANE = MinMaxScaler()
data_all['DANE_Year_normalized'] = scaler_DDANE.fit_transform(data_all[['DANE_Year_labeled']])

# Descriptive statistics of deforestation (treatment)
std_deforestation = data_all['Deforestation_t'].std()
median_deforestation = data_all['Deforestation_t'].median()
mean_deforestation = data_all['Deforestation_t'].mean()

print(f"\n{'='*60}")
print("TREATMENT STATISTICS (Deforestation_t)")
print(f"{'='*60}")
print(f"Mean: {mean_deforestation:.4f}")
print(f"Median: {median_deforestation:.4f}")
print(f"Standard Deviation: {std_deforestation:.4f}")
print(f"{'='*60}\n")

# Standardization of continuous variables (Z-score)
scaler_std = StandardScaler()
vars_to_standardize = ['MPI', 'Forest_tm1', 'HFP_t', 'Illegal_mining_t',
                       'Soil_Moisture', 'Temperature', 'Precipitation',
                       'Vectors', 'Fire_t', 'Coca_t', 'Deforestation_t']

for var in vars_to_standardize:
    if var in data_all.columns:
        data_all[var] = scaler_std.fit_transform(data_all[[var]])

# Final standardized dataset
data_std = data_all[['DANE_normalized', 'DANE_Year_normalized', 'Year',
               'MPI', 'Forest_tm1', 'HFP_t', 'Illegal_mining_t', 'Soil_Moisture',
               'Temperature', 'Precipitation', 'Vectors', 'Fire_t',
               'Coca_t', 'Deforestation_t', 'Excess_cases_tp1']].copy()

# Ensure correct temporal order
data_std = data_std.sort_values(
    by=['DANE_normalized', 'DANE_Year_normalized']
).reset_index(drop=True)

data_std = data_std.dropna()

print(f"Final dataset for causal analysis: {data_std.shape}")
print(f"\nFirst 5 observations:")
print(data_std.head())

#%%

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



#%%

print("\n" + "=" * 70)
print("IDENTIFICATION OF THE CAUSAL ESTIMAND")
print("=" * 70)

identified_estimand_deforest = model_deforestation.identify_effect(
    proceed_when_unidentifiable=True
)
print(identified_estimand_deforest)

#%%

from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV

# ============================================================================
# WRAPPER FOR PROBABILISTIC CLASSIFIERS
# ============================================================================

class ProbClassifierWrapper(BaseEstimator):
    """
    Wrapper for sklearn classifiers that returns probabilities in predict().

    For binary outcomes, EconML expects model_y.predict(X) to return
    E[Y|X] as continuous probabilities, not discrete classes.
    """
    def __init__(self, base_clf=None, calibrate=True, random_state=123):
        if base_clf is None:
            base_clf = HistGradientBoostingClassifier(
                max_iter=200,
                #n_jobs=1,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=1 
            )
        self.base_clf = base_clf
        self.calibrate = calibrate
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X, y, **kwargs):
        """Fits the base classifier (with or without calibration)"""
        y = np.asarray(y).ravel()

        if self.calibrate:
            self.model_ = CalibratedClassifierCV(
                estimator=clone(self.base_clf),
                cv=3
            )
            self.model_.fit(X, y)
        else:
            self.model_ = clone(self.base_clf)
            self.model_.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X):
        """Returns positive-class probabilities (P(Y=1|X))"""
        if not self._is_fitted:
            raise ValueError("ProbClassifierWrapper must be fitted before predict()")
        return self.model_.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        """Returns the full probability matrix"""
        if not self._is_fitted:
            raise ValueError("ProbClassifierWrapper must be fitted before predict_proba()")
        return self.model_.predict_proba(X)


# Required imports
from econml.score import RScorer
import numpy as np

# ProbClassifierWrapper is assumed to be defined and available in a previous cell.

# Define the fixed parameters for XGBoost
xgb_fixed_params = {
    "random_state": 123,
    "learning_rate": 0.05,
    #"reg_lambda": 'l2_regularization',
    #"alpha": 0.001,
}


#%%




# Define the model configurations to test
model_configs = [
    {"name": "Model 1", "max_iter": 10, "max_depth": 2},
    {"name": "Model 2", "max_iter": 10, "max_depth": 3},
    {"name": "Model 3", "max_iter": 50, "max_depth": 2},
    {"name": "Model 4", "max_iter": 50, "max_depth": 3},
    {"name": "Model 5", "max_iter": 100, "max_depth": 2},
    {"name": "Model 6", "max_iter": 100, "max_depth": 3},
    {"name": "Model 7", "max_iter": 150, "max_depth": 2},
    {"name": "Model 8", "max_iter": 150, "max_depth": 3},
    {"name": "Model 9", "max_iter": 200, "max_depth": 2},
    {"name": "Model 10", "max_iter": 200, "max_depth": 3}, 

]
    
#%%


# Prepare data
Y = data_std['Excess_cases_tp1'].values.astype(int)
T = data_std['Deforestation_t'].values
municipality_groups = data_std['DANE_normalized'].values

X = data_std[['Forest_tm1']].values

W = data_std[
    ['Coca_t', 'Fire_t', 'Illegal_mining_t', 'MPI', 'Forest_tm1', 'HFP_t', 'Temperature']
].values

# ---------------------------------------------------------
# Build a SINGLE mask of complete observations
# ---------------------------------------------------------

mask_ok = (
    np.isfinite(Y) &
    np.isfinite(T) &
    np.isfinite(X).all(axis=1) &
    np.isfinite(W).all(axis=1) &
    np.isfinite(municipality_groups)
)

# Apply EXACTLY the same mask to all objects
Y = Y[mask_ok]
T = T[mask_ok]
X = X[mask_ok, :]
W = W[mask_ok, :]
municipality_groups = municipality_groups[mask_ok]

print(f"Shapes after filtering:")
print(f"Y      = {Y.shape}")
print(f"T      = {T.shape}")
print(f"X      = {X.shape}")
print(f"W      = {W.shape}")
print(f"groups = {municipality_groups.shape}")

assert len(Y) == len(T)
assert len(Y) == len(X)
assert len(Y) == len(W)
assert len(Y) == len(municipality_groups)

#%%

print("\n============================================================")
print("Computing R-Score for multiple DML models")
print("============================================================")

# ---> SOLUTION: Initialize the list before the loop <---
results_rscore = []

for config in model_configs:
    model_name = config["name"]
    n_est = config["max_iter"]
    depth = config["max_depth"]

    print(f"\n--- Evaluating {model_name} (max_iter={n_est}, max_depth={depth}) ---")

    # 1) Define the nuisance models
    model_y_nuisance = HistGradientBoostingClassifier(
        max_iter=n_est,
        max_depth=depth,
        **xgb_fixed_params,
    )
    # Wrap for probability prediction
    model_y_wrapped = ProbClassifierWrapper(
        base_clf=model_y_nuisance, 
        calibrate=True, 
        random_state=xgb_fixed_params["random_state"]
    )

    model_t_nuisance = HistGradientBoostingRegressor(
        max_iter=n_est,
        max_depth=depth,
        **xgb_fixed_params,
    )

    # 2) Instantiate and fit the causal DML model
    dml_model = DML(
        model_y=model_y_wrapped,
        model_t=model_t_nuisance,
        model_final=LassoCV(
            alphas=[0.0001, 0.001, 0.005, 0.05, 0.01, 0.1],
            fit_intercept=False,
            max_iter=50000,
            tol=1e-3,
            cv=3,
        ),
        # CATE linear: no polynomial featurizer (consistent
        # with the main DML script). model_final = LassoCV on X.
        discrete_outcome=True,
        discrete_treatment=False,
        cv=GroupKFold(n_splits=3),
        random_state=xgb_fixed_params["random_state"]
    )
    
    print("Fitting DML...")
    dml_model.fit(Y=Y, T=T, X=X, W=W, groups=municipality_groups)
    print("DML fitted.")

    # 3) Compute R-Score using the dml_model .score() method
    print("Computing R-Score with dml_model.score()...")
    r_score_value = dml_model.score(Y=Y, T=T, X=X, W=W)

    # results_rscore now exists and append() will work correctly
    results_rscore.append((model_name, r_score_value))
    print(f"  R-Score for {model_name}: {r_score_value:.6f}")

print("\n============================================================")
print("Final R-Score Results:")
print("============================================================")
for name, score in results_rscore:
    print(f"  {name}: {score:.6f}")
print("============================================================")


#%%




