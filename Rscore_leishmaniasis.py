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

# Compatibility with modern numpy versions
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

# --- Data Import & Preprocessing ---
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load CSV file from Drive
file_path = '/content/drive/MyDrive/data_final_15_sep.csv'
data_all = pd.read_csv(file_path, encoding='latin-1')

# Remove rows with missing values
data_all = data_all.dropna()
print(f"Dimensions after dropna: {data_all.shape}")

# Drop columns unnecessary for causal analysis
# Note: These are either redundant or non-causal descriptors
columnas_to_drop = ['Year', 'Altitude', 'Forest', 'cases', 'total_pop',
                     'expected', 'sir', 'Excess_cases']
data_all.drop(columns=columnas_to_drop, inplace=True, errors='ignore')

# 1. Label Encoding for DANE (Municipality identifier - Spatial component)
le = LabelEncoder()
data_all['DANE_labeled'] = le.fit_transform(data_all['DANE'])
scaler = MinMaxScaler()
data_all['DANE_normalized'] = scaler.fit_transform(data_all[['DANE_labeled']])

# 2. Label Encoding for DANE_Year (Time-series identifier - Temporal component)
le_year = LabelEncoder()
data_all['DANE_Year_labeled'] = le_year.fit_transform(data_all['DANE_year'])
scaler_DDANE = MinMaxScaler()
data_all['DANE_Year_normalized'] = scaler_DDANE.fit_transform(data_all[['DANE_Year_labeled']])

# Descriptive statistics for Deforestation (Treatment Variable)
std_deforestation = data_all['Deforestation_t'].std()
median_deforestation = data_all['Deforestation_t'].median()
mean_deforestation = data_all['Deforestation_t'].mean()

print(f"\n{'='*60}")
print("TREATMENT STATISTICS (Deforestation_t)")
print(f"{'='*60}")
print(f"Mean: {mean_deforestation:.4f}")
print(f"Median: {median_deforestation:.4f}")
print(f"Std Dev: {std_deforestation:.4f}")
print(f"{'='*60}\n")

# Standardization of continuous variables (Z-score normalization)
scaler_std = StandardScaler()
vars_to_standardize = ['MPI', 'Forest_tm1', 'HFP_t', 'Illegal_mining_t',
                       'Soil_Moisture', 'Temperature', 'Precipitation',
                       'Vectors', 'Fire_t', 'Coca_t', 'Deforestation_t']

for var in vars_to_standardize:
    if var in data_all.columns:
        data_all[var] = scaler_std.fit_transform(data_all[[var]])

# Final standardized dataset for Causal Inference
df = data_all[['DANE_normalized', 'DANE_Year_normalized',
               'MPI', 'Forest_tm1', 'HFP_t', 'Illegal_mining_t', 'Soil_Moisture',
               'Temperature', 'Precipitation', 'Vectors', 'Fire_t',
               'Coca_t', 'Deforestation_t', 'Excess_cases_tp1']].copy()

print(f"Final dataset for causal analysis: {df.shape}")
print(f"\nFirst 5 observations:")
print(df.head())

#%%
class ProbClassifierWrapper(BaseEstimator):
    """
    Wrapper for sklearn classifiers to return probabilities in predict().
    
    For binary outcomes, EconML expects model_y.predict(X) to return
    E[Y|X] as continuous probabilities rather than discrete classes.
    """
    def __init__(self, base_clf=None, calibrate=True, random_state=123):
        if base_clf is None:
            base_clf = RandomForestClassifier(
                n_estimators=200, 
                n_jobs=1, 
                random_state=random_state, 
                class_weight='balanced'
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
        """Returns positive class probabilities (P(Y=1|X))"""
        if not self._is_fitted:
            raise ValueError("ProbClassifierWrapper must be fitted before calling predict()")
        return self.model_.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        """Returns full probability matrix"""
        if not self._is_fitted:
            raise ValueError("ProbClassifierWrapper must be fitted before calling predict_proba()")
        return self.model_.predict_proba(X)
    
#%%
def prepare_data_vectors(df):
    """
    Prepares data matrices according to specific user requirements.
    y: Outcome (Binary)
    t: Treatment (Continuous)
    W: Controls/Confounders (Backward Path)
    X: Features for Heterogeneity (CATE)
    """
    y = df['Excess_cases_tp1'].astype(int).values.ravel()
    t = df['Deforestation_t'].astype(float).values.ravel()
    W = df[['Coca_t', 'Fire_t', 'Illegal_mining_t', 'MPI', 'Forest_tm1', 'HFP_t', 'Temperature']].values
    X = df[['Forest_tm1', 'DANE_normalized', 'DANE_Year_normalized']].values
    
    return y, t, W, X

# ============================================================================
# NUISANCE MODELS DIAGNOSTICS
# ============================================================================

def nuisance_diagnostics(y_tr, t_tr, W_tr, X_tr, y_val, t_val, W_val, X_val, random_state=123):
    """
    Diagnoses the quality of nuisance models (First-stage estimation).
    Calculates residuals for the orthogonalization process.
    """
    Z_tr = np.hstack([X_tr, W_tr])
    Z_val = np.hstack([X_val, W_val])

    # Model Y: Probabilistic classifier for binary outcome
    model_y = ProbClassifierWrapper(
        RandomForestClassifier(
            n_estimators=200, 
            n_jobs=1, 
            class_weight='balanced', 
            random_state=random_state
        ),
        calibrate=True, 
        random_state=random_state
    )
    model_y.fit(Z_tr, y_tr)
    y_pred_val = model_y.predict(Z_val)

    # Model T: Regressor for continuous treatment
    model_t = RandomForestRegressor(
        n_estimators=200, 
        n_jobs=1, 
        random_state=random_state
    )
    model_t.fit(Z_tr, t_tr)
    t_pred_val = model_t.predict(Z_val)

    # Residuals (Orthogonalized components)
    y_res = y_val - y_pred_val
    t_res = t_val - t_pred_val

    diagnostics = {
        'y_pred_val_mean': np.mean(y_pred_val),
        'y_res_var': np.var(y_res),
        't_pred_val_mean': np.mean(t_pred_val),
        't_res_var': np.var(t_res),
        'y_t_res_corr': np.corrcoef(y_res, t_res)[0, 1] if len(y_res) > 1 else np.nan
    }
    
    return diagnostics, y_pred_val, t_pred_val, y_res, t_res

#%%

def make_sparselinear_models(random_state=123):
    """
    Generates SparseLinearDML models with a hyperparameter grid for nuisance models.
    
    CRITICAL: n_estimators and max_depth are applied ONLY to the nuisance models
    (RandomForest), NOT directly to the SparseLinearDML estimator.
    """
    models = []
    
    # Hyperparameter grid for Random Forest (Nuisance models)
    hyperparameter_grid = [
        (3400, 35), (3600, 33), (3700, 31),
        (3500, 40), (3400, 30), (3600, 32)
    ]
    
    print(f"Generating {len(hyperparameter_grid)} model configurations...")
    
    for n_est, depth in hyperparameter_grid:
        name = f"SparseLinearDML_n{n_est}_d{depth}"
        
        # =====================================================================
        # NUISANCE MODEL Y (Outcome): Classifier for binary target
        # =====================================================================
        base_clf_y = RandomForestClassifier(
            n_estimators=n_est,          
            max_depth=depth,              
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',          # Additional regularization
            n_jobs=1,
            class_weight='balanced',
            random_state=random_state
        )
        
        model_y_inst = ProbClassifierWrapper(
            base_clf=base_clf_y,
            calibrate=True,
            random_state=random_state
        )
        
        # =====================================================================
        # NUISANCE MODEL T (Treatment): Regressor for continuous variable
        # =====================================================================
        model_t_inst = RandomForestRegressor(
            n_estimators=n_est,          
            max_depth=depth,              
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=1,
            random_state=random_state
        )
        
        # =====================================================================
        # CAUSAL ESTIMATOR: SparseLinearDML
        # =====================================================================
        try:
            sparselinear_dml = SparseLinearDML(
                model_y=model_y_inst,              
                model_t=model_t_inst,              
                discrete_outcome=True,             
                discrete_treatment=False,          
                fit_cate_intercept=True,           
                alpha='auto',                       
                max_iter=50000,                    
                tol=1e-4,                          
                cv=5,                              
                random_state=random_state          
            )
            
            models.append((name, sparselinear_dml))
            print(f"  ✓ {name} successfully configured")
            
        except TypeError as e:
            print(f"  ❌ ERROR in {name}: {e}")
            raise
    
    print(f"\n✓ Total models generated: {len(models)}")
    return models    

#%%

# ============================================================================
# MODEL EVALUATION WITH RSCORER (UPDATED API)
# ============================================================================

def fit_and_evaluate_with_rscore(df, models_to_try, random_state=123):
    """
    Fits multiple DML models and evaluates them using RScorer.
    Compatible with EconML >= 0.13
    """
    print(f"\n{'='*80}")
    print("STARTING MODEL EVALUATION WITH RSCORER")
    print(f"{'='*80}")

    # Prepare data
    y, t, W, X = prepare_data_vectors(df)

    n = len(y)
    n_positives = int(y.sum())
    prevalence = n_positives / n
    var_t = np.var(t)

    print(f"\nDataset Statistics:")
    print(f"  N observations: {n}")
    print(f"  Positive cases (Excess_cases_tp1=1): {n_positives} ({prevalence:.2%})")
    print(f"  Treatment variance (Deforestation_t): {var_t:.6f}")

    # Stratified train/validation split
    print(f"\nPerforming stratified split (60% train, 40% validation)...")

    try:
        X_tr, X_val, t_tr, t_val, y_tr, y_val, W_tr, W_val = train_test_split(
            X, t, y, W,
            test_size=0.4,
            random_state=random_state,
            stratify=y
        )
    except ValueError:
        print("⚠️ Stratification not possible. Using random split.")
        X_tr, X_val, t_tr, t_val, y_tr, y_val, W_tr, W_val = train_test_split(
            X, t, y, W,
            test_size=0.4,
            random_state=random_state
        )

    # Nuisance models diagnostics
    diag, _, _, _, _ = nuisance_diagnostics(
        y_tr, t_tr, W_tr, X_tr,
        y_val, t_val, W_val, X_val,
        random_state=random_state
    )

    # Fit models in parallel
    def fit_single_model(name, model):
        try:
            model.fit(Y=y_tr, T=t_tr, X=X_tr, W=W_tr)
            return (name, model, None)
        except Exception as e:
            return (name, None, str(e))

    results = Parallel(n_jobs=-1, verbose=5, backend="threading")(
        delayed(fit_single_model)(name, mdl) for name, mdl in models_to_try
    )

    fitted_models = []
    failed_models = []
    rscores = []

    for name, mdl, error in results:
        if error is not None:
            failed_models.append((name, error))
        else:
            fitted_models.append((name, mdl))
            # Calculate RScore for validation
            try:
                rscore = mdl.score(Y=y_val, T=t_val, X=X_val, W=W_val)
                rscores.append((name, rscore))
            except:
                rscores.append((name, np.nan))

    best_model = max([(n, s) for n, s in rscores if not np.isnan(s)], key=lambda x: x[1], default=None)

    return {
        'n': n, 'prevalence': prevalence, 'var_treatment': var_t,
        'nuisance_diagnostics': diag, 'fitted_models': fitted_models,
        'rscores': rscores, 'best_model': best_model, 'failed_models': failed_models,
        'X_tr': X_tr, 'X_val': X_val, 'y_tr': y_tr, 'y_val': y_val, 
        't_tr': t_tr, 't_val': t_val, 'W_tr': W_tr, 'W_val': W_val
    }

#%%
# ============================================================================
# MAIN EXECUTION SCRIPT
# ============================================================================

print("\n" + "="*80)
print("CAUSAL ANALYSIS: DEFORESTATION → LEISHMANIASIS")
print("Method: Double Machine Learning (SparseLinearDML)")
print("Evaluation: RScorer")
print("="*80)

# Generate candidate models
models_to_try = make_sparselinear_models(random_state=seed)
analysis_results = fit_and_evaluate_with_rscore(df, models_to_try, random_state=seed)

# --- Summary and Results Interpretation ---
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

if analysis_results['best_model'] is not None:
    best_name, best_score = analysis_results['best_model']
    print(f"🏆 BEST MODEL: {best_name} | R² Score: {best_score:.6f}")
    
    # Retrieve best model object
    best_mdl_obj = next(mdl for name, mdl in analysis_results['fitted_models'] if name == best_name)
    
    # Estimate Average Treatment Effect (ATE)
    ate_inf = best_mdl_obj.ate_inference(X=analysis_results['X_val'])
    ate_val = best_mdl_obj.ate(X=analysis_results['X_val'])
    ate_ci = ate_inf.conf_int_mean(alpha=0.05)

    print(f"\n📊 AVERAGE TREATMENT EFFECT (ATE):")
    print(f"  Estimate: {ate_val[0]:.6f}")
    print(f"  95% CI: [{ate_ci[0][0]:.6f}, {ate_ci[1][0]:.6f}]")

    # Conditional Average Treatment Effects (CATE) Analysis
    cate_preds = best_mdl_obj.effect(X=analysis_results['X_val'])
    print(f"\n📈 CATE DISTRIBUTION:")
    print(f"  Mean: {np.mean(cate_preds):.6f}")
    print(f"  Range: [{np.min(cate_preds):.6f}, {np.max(cate_preds):.6f}]")
else:
    print("⚠️ No valid best model identified.")

print("\n" + "="*80)
print("--- SCRIPT COMPLETED SUCCESSFULLY ---")
print("="*80)