# LIBRARY IMPORTATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from itertools import combinations

import joblib

RANDOM_STATE = 42
print('library importation step done !')

# DATA IMPORTATION
data = pd.read_csv('data/InsuranceLeadData.csv')
print(data.head())
print('data importation step done !')

# DATA PREPROCESSING
# rename data columns
new_columns = ['id', 'city_code', 'region_code', 'accomodation_type', 'reco_insurance_type', 'upper_age', 'lower_age', 'is_spouse', 'health_indicator', 'holding_policy_duration'\
               , 'holding_policy_type', 'reco_policy_cat', 'reco_policy_premium', 'response']
data.columns = new_columns
data.head()

# check and remove duplicated rows
data.drop_duplicates(inplace = True)
data.reset_index(drop = True, inplace = True)

# transform datatype
data['region_code'] = data['region_code'].astype('str')
data['reco_policy_cat'] = data['reco_policy_cat'].astype('str')

# check null values, get percentage of null values
data.isnull().sum()/len(data)

# fix missing value
data['health_indicator'] = data['health_indicator'].fillna(data['health_indicator'].mode()[0])

# drop columns with high percentage of missing value
data.drop(columns = ['holding_policy_duration', 'holding_policy_type'], inplace = True)

# drop irrelevant column
data.drop(columns = ['id'], inplace = True)

# Split data into train and test parts
df_full_train, df_test = train_test_split(data, test_size = 0.3, random_state = RANDOM_STATE)
print('data preprocessing step done !')

# DATA EXPLORATION 
# use log transformation to fix right skewed data distribution
df_full_train['reco_policy_premium_log'] = np.log1p(df_full_train['reco_policy_premium'])
df_full_train.drop(columns = ['reco_policy_premium'], inplace = True)

# remove outliers for numerical features
numerical_columns = df_full_train.select_dtypes(include = ['int', 'float']).columns
numerical_columns = [item for item in numerical_columns if item != 'response']
categorical_columns = df_full_train.select_dtypes(include = 'object').columns.tolist()
numerical_columns, categorical_columns

list_q99_outliers = []
list_iqr_outliers = []
for num in numerical_columns:
    q99 = np.quantile(df_full_train[num], .99)
    q99_outliers= df_full_train[df_full_train[num] > q99].index.tolist()
    list_q99_outliers.append([num, q99_outliers, len(q99_outliers)])
    
    q1 = np.quantile(df_full_train[num], .25)
    q3 = np.quantile(df_full_train[num], .25)
    iqr = q3 - q1
    iqr_lower_fence = q1 - 1.5*iqr
    iqr_higher_fence = q3 + 1.5*iqr
    iqr_outliers = df_full_train[(df_full_train[num] < iqr_lower_fence) & (df_full_train[num] > iqr_higher_fence)].index.tolist()
    list_iqr_outliers.append([num, iqr_outliers, len(iqr_outliers)])

df_iqr_outliers = pd.DataFrame(list_iqr_outliers, columns = ['feature', 'index', 'nb_outliers'])
print(df_iqr_outliers.head())    
df_outliers_q99 = pd.DataFrame(list_q99_outliers, columns = ['feature', 'index', 'nb_outliers'])
print(df_outliers_q99.head())

# remove outliers and get cleaned data for training model
df_full_train_cleaned = df_full_train.drop(index=df_outliers_q99.query('feature == "reco_policy_premium_log"').iloc[:,1].tolist()[0])
print('data exploration step done !')

# DATA PREPARATION FOR TRAINING MODEL
# split data into train and validation data
df_train, df_validation = train_test_split(df_full_train_cleaned, test_size=0.3, random_state = RANDOM_STATE)
df_train.reset_index(drop = True, inplace = True)
df_validation.reset_index(drop = True, inplace = True)

# create new features
df_reco_policy_cat = df_train.groupby(['reco_policy_cat'], as_index=False)['response'].mean()
df_reco_policy_cat.sort_values(by = ['response'], ascending = False)
list_reco_policy_cat = df_reco_policy_cat.query('response >= 0.25')['reco_policy_cat'].unique().tolist()
# ====> reco_policy_cat_mapping: response_mean >= 0.25 -> health_indicator, response_mean < 0.25 -> autre

df_city_code = df_train.groupby(['city_code'], as_index=False)['response'].mean()
list_top_city = df_city_code.query('response >= 0.25')['city_code'].unique().tolist()
# ====> top_city : response_mean >= 0.25 -> city_code , response_mean < 0.25 -> autre

df_train['city_code_group'] = np.where(df_train['city_code'].isin(list_top_city), df_train['city_code'], 'autre')
df_train['reco_policy_cat_group'] = np.where(df_train['reco_policy_cat'].isin(list_reco_policy_cat), df_train['reco_policy_cat'], 'autre')

df_validation['city_code_group'] = np.where(df_validation['city_code'].isin(list_top_city), df_validation['city_code'], 'autre')
df_validation['reco_policy_cat_group'] = np.where(df_validation['reco_policy_cat'].isin(list_reco_policy_cat), df_validation['reco_policy_cat'], 'autre')

df_test['reco_policy_premium_log'] = np.log1p(df_test['reco_policy_premium'])
df_test['city_code_group'] = np.where(df_test['city_code'].isin(list_top_city), df_test['city_code'], 'autre')
df_test['reco_policy_cat_group'] = np.where(df_test['reco_policy_cat'].isin(list_reco_policy_cat), df_test['reco_policy_cat'], 'autre')
print('data preparation step done !')

# BUILD BASELINE MODEL
# use all features for baseline model
selected_features = ['reco_policy_premium_log',  'upper_age', 'lower_age', 'city_code_group', 'is_spouse', 'health_indicator', 'reco_policy_cat_group', 'accomodation_type', 'reco_insurance_type', 'response'] 

# create training, validation and test dataset
df_train = df_train[selected_features]
df_validation = df_validation[selected_features]
df_test = df_test[selected_features]

X_train = df_train.drop(columns = ['response'])
y_train = df_train['response']
X_validation = df_validation.drop(columns = ['response'])
y_validation = df_validation['response']
X_test = df_test.drop(columns = ['response'])
y_test = df_test['response']

# get best parameters for choosen algoirthm
dt_max_depth = 10
dt_min_samples_leaf = 30
rf_n_estimators = 100
rf_max_depth = 10
lr_c = 0.100
xgb_eta = 0.01
xgb_n_estimators = 100
xgb_max_depth = 5

# init baseline models
dt_model = DecisionTreeClassifier(random_state = RANDOM_STATE, max_depth = dt_max_depth, min_samples_leaf= dt_min_samples_leaf)
lr_model = LogisticRegression(C = lr_c, max_iter=1000)
rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators= rf_n_estimators, max_depth = rf_max_depth, )
xgb_model = xgb.XGBClassifier(
                eta = xgb_eta, 
                max_depth = xgb_max_depth,
                min_child_weight = 1,
                objective = 'binary:logistic',
                nthread = 12,
                random_state = RANDOM_STATE,
                verbosity = 1,
                n_estimators = xgb_n_estimators
            )

stratifiedkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
df_full_train_to_model = pd.concat([df_train, df_validation,])
X_full_train = df_full_train_to_model.drop(columns = ['response'])
y_full_train = df_full_train_to_model['response']

# set transformers for preprocessing
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop = 'first', handle_unknown='ignore', sparse_output = False))
])

scores_cv = []
for train_idx, val_idx in stratifiedkfold.split(X_full_train, y_full_train):

    X_train_ = X_full_train.iloc[train_idx]
    y_train_ = y_full_train.iloc[train_idx]
    
    X_validation_ = X_full_train.iloc[val_idx]
    y_validation_ = y_full_train.iloc[val_idx]
    
    numerical_selected_features_cv = X_train_.select_dtypes(include = ['integer', 'float']).columns.tolist()
    categorical_selected_features_cv = X_train_.select_dtypes(include = ['object']).columns.tolist()

    preprocessor_cv = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_selected_features_cv),
            ('cat', categorical_transformer, categorical_selected_features_cv),
        ])

    pipeline_cv = Pipeline(steps=[('preprocessor', preprocessor_cv)])
    pipeline_cv.fit(X_train_, y_train_)
    
    all_columns_cv = pipeline_cv.named_steps['preprocessor'].transformers_[0][2] + pipeline_cv.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out().tolist()
    X_train_transformed_cv = pd.DataFrame(pipeline_cv.named_steps['preprocessor'].transform(X_train_), columns = all_columns_cv)
    X_validation_transformed_cv = pd.DataFrame(pipeline_cv.named_steps['preprocessor'].transform(X_validation_), columns = all_columns_cv)
    X_test_transformed_cv =  pd.DataFrame(pipeline_cv.named_steps['preprocessor'].transform(X_test), columns = all_columns_cv)
    
    dt_model.fit(X_train_transformed_cv, y_train_)
    y_train_pred = dt_model.predict(X_train_transformed_cv)
    y_validation_pred = dt_model.predict(X_validation_transformed_cv)
    y_test_pred = dt_model.predict(X_test_transformed_cv)
    try: 
        val_score = roc_auc_score(y_validation_pred, y_validation_)
    except:
        val_score = None
    try:
        test_score = roc_auc_score(y_test_pred, y_test)
    except:
        test_score = None
    finally:
        scores_cv.append(('dt_best', val_score, test_score))        
        
    lr_model.fit(X_train_transformed_cv, y_train_)
    y_train_pred = lr_model.predict(X_train_transformed_cv)
    y_validation_pred = lr_model.predict(X_validation_transformed_cv)
    y_test_pred = lr_model.predict(X_test_transformed_cv)
    try: 
        val_score = roc_auc_score(y_validation_pred, y_validation_)
    except:
        val_score = None
    try:
        test_score = roc_auc_score(y_test_pred, y_test)
    except:
        test_score = None
    finally:
        scores_cv.append(('lr_best', val_score, test_score))        
    
    rf_model.fit(X_train_transformed_cv, y_train_)
    y_train_pred = rf_model.predict(X_train_transformed_cv)
    y_validation_pred = rf_model.predict(X_validation_transformed_cv)    
    y_test_pred = rf_model.predict(X_test_transformed_cv)
    try: 
        val_score = roc_auc_score(y_validation_pred, y_validation_)
    except:
        val_score = None
    try:
        test_score = roc_auc_score(y_test_pred, y_test)
    except:
        test_score = None
    finally:
        scores_cv.append(('rf_best', val_score, test_score))  

    xgb_model.fit(X_train_transformed_cv, y_train_)
    y_train_pred = xgb_model.predict(X_train_transformed_cv)
    y_validation_pred = xgb_model.predict(X_validation_transformed_cv)
    y_test_pred = xgb_model.predict(X_test_transformed_cv)
    try: 
        val_score = roc_auc_score(y_validation_pred, y_validation_)
    except:
        val_score = None
    try:
        test_score = roc_auc_score(y_test_pred, y_test)
    except:
        test_score = None
    finally:
        scores_cv.append(('xgb_best', val_score, test_score))

       
df_scores_cv = pd.DataFrame(scores_cv, columns = ['model', 'auc_roc_score_val', 'auc_roc_score_test'])
print(df_scores_cv.groupby(['model'], as_index=False).agg({'auc_roc_score_val' : 'mean', 'auc_roc_score_test' : 'mean'}))

# choose and train xgboost model as baseline model
pipeline_baseline_model = Pipeline(steps=[
    ('preprocessor', preprocessor_cv),
    ('classifier', xgb_model)
])

# fit baseline model/pipeline
pipeline_baseline_model.fit(X_full_train, y_full_train)

# evaluate baseline model
y_train_pred = pipeline_baseline_model.predict(X_full_train)
y_test_pred = pipeline_baseline_model.predict(X_test)
print(f'XGBoost baseline model ROC AUC score on training data : {roc_auc_score(y_train_pred, y_full_train)}')
print(f'XGBoost baseline model ROC AUC score on test data : {roc_auc_score(y_test_pred, y_test)}')
print('building baseline model step done !')

# IMPROVE BASELINE MODEL
# use grid search for selecting best features
numerical_features_gs = ['reco_policy_premium_log', 'upper_age', 'lower_age',]
categorical_features_gs = [
   'city_code_group','is_spouse', 'health_indicator', 'reco_policy_cat_group','accomodation_type', 'reco_insurance_type'
]

# Create feature combinations
all_features = numerical_features_gs + categorical_features_gs
feature_combinations = []
for r in range(4, len(all_features) + 1):
    feature_combinations.extend(combinations(all_features, r))
feature_combinations = [list(comb) for comb in feature_combinations]    

# Create a custom transformer for feature selection
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]
    
# Combine preprocessing steps
preprocessor_gs = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('cat', categorical_transformer, make_column_selector(dtype_include=['object']))
    ])
pipeline_gs = Pipeline([
    ('selector', ColumnSelector(columns=[])), 
    ('preprocessor',  preprocessor_gs),
    ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, n_estimators= rf_n_estimators, max_depth = rf_max_depth, ))     
])

# Paramètres pour le modèle KMeans
param_grid = {
    'selector__columns': feature_combinations,  # Combinaisons de variables
}

grid_search = GridSearchCV(
    pipeline_gs,
    param_grid=param_grid,
    scoring= 'roc_auc',
    refit=True,
    cv=5,  
    verbose=3,
)

# fit grid_search
grid_search_results  = grid_search.fit(X_full_train, y_full_train)

# get grid search model
grid_search_results_df = pd.DataFrame(grid_search_results.cv_results_)

# save grid search model for further analysis and features combination selection
grid_search_results_df.to_csv('data/grid_search_results_df.csv', index=False, encoding = 'utf-8-sig', mode = 'w')

# set best features combinations
best_features = ['reco_policy_premium_log', 'city_code_group', 'health_indicator', 'reco_policy_cat_group', 'accomodation_type']

# filter training data
X_full_train_selected = X_full_train[best_features]

preprocessor_final = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, X_full_train_selected.select_dtypes(include = ['integer', 'float']).columns.tolist()),
        ('cat', categorical_transformer, X_full_train_selected.select_dtypes(include = ['object']).columns.tolist()),
])

pipeline_best_model = Pipeline(steps=[
    ('preprocessor', preprocessor_final),
    ('classifier', xgb_model)
])

pipeline_best_model.fit(X_full_train_selected, y_full_train)

# Save final model (best model)
with open('model/pipeline_best_model.pkl', 'wb') as f:
    joblib.dump(pipeline_best_model, f)
    
# load pipeline to a file
with open('model/pipeline_best_model.pkl', 'rb') as f:
    loaded_pipeline = joblib.load(f)
    
# Evaluate final model
y_train_pred = loaded_pipeline.predict(X_full_train)
y_test_pred = loaded_pipeline.predict(X_test)
print(f'XGBoost best model ROC AUC score on training data : {roc_auc_score(y_train_pred, y_full_train)}')
print(f'XGBoost best model ROC AUC score on test data : {roc_auc_score(y_test_pred, y_test)}')
print('building final model step done !')