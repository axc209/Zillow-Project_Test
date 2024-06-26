# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:48:55 2024

@author: 1043100
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotnine as pltn
import statsmodels.api as sm
from statsmodels.formula.api import glm
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.nonparametric as npar
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


df = pd.read_excel('Zillow_Details.xlsx')

columns_to_drop = [
    "abbreviatedAddress", "brokerIdDimension", "brokerageName", "country",
    "datePostedString", "datePriceChanged", "dateSold", "dateSoldString",
    "hasVRModel", "latitude", "listPriceLow", "livingAreaUnitsShort",
    "longitude", "priceChangeDate", "priceChangeDateString", "rentalApplicationsAcceptedType",
    "restimateHighPercent", "restimateLowPercent", "taxAssessedYear", "timeZone",
    "tourViewCount", "zestimate", "zestimateHighPercent", "zestimateLowPercent",
    "address.subdivision", "downPaymentAssistance.maxAssistance", "downPaymentAssistance.resultCount",
    "foreclosureTypes.isBankOwned", "listing_agent.display_name", "listing_agent.first_name",
    "listing_agent.rating_average", "listing_agent.recent_sales", "listing_agent.review_count",
    "resoFacts.availabilityDate", "resoFacts.buildingAreaSource", "resoFacts.buyerAgencyCompensation",
    "resoFacts.buyerAgencyCompensationType", "resoFacts.cityRegion", "resoFacts.daysOnZillow",
    "resoFacts.electric", "resoFacts.entryLocation", "resoFacts.hoaFee", "resoFacts.homeType",
    "resoFacts.leaseTerm", "resoFacts.listingTerms", "resoFacts.livingArea", "resoFacts.livingAreaRangeUnits",
    "resoFacts.lotSize", "resoFacts.lotSizeDimensions", "resoFacts.middleOrJuniorSchool",
    "resoFacts.middleOrJuniorSchoolDistrict", "resoFacts.onMarketDate", "resoFacts.openParkingSpaces",
    "resoFacts.otherParking", "resoFacts.ownership", "resoFacts.specialListingConditions",
    "resoFacts.storiesTotal", "resoFacts.subdivisionName", "resoFacts.topography", "resoFacts.yearBuilt",
    "resoFacts.yearBuiltEffective", "priceHistory_date", "priceHistory_event", "priceHistory_price",
    "priceHistory_pricePerSquareFoot", "priceHistory_showCountyLink", "priceHistory_source", "priceHistory_time",
    "schools_grades", "schools_level", "schools_name", "sellingSoon_0", "taxHistory_taxIncreaseRate",
    "taxHistory_taxPaid", "taxHistory_time", "taxHistory_value", "taxHistory_valueIncreaseRate",
    "resoFacts.atAGlanceFacts_factLabel", "resoFacts.rooms_roomArea", "resoFacts.rooms_roomDimensions",
    "associationFee_text", "associationFee_frequency"
]

# Drop the specified columns
df = df.drop(columns=columns_to_drop, errors='ignore')

df.to_excel('Regression Zillow.xlsx',index=False)

# Convert True/False values to 1/0 for all columns
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].map({True: 1, False: 0})
        
# Get the data types of all columns
data_types = df.dtypes

# Filter columns that have 'object' data type
object_columns = data_types[data_types == 'object'].index.tolist()
del df['ZPID']

# Convert 'city' and 'county' columns into dummy variables
df = pd.get_dummies(df, columns=object_columns)
# Convert again True/False values to 1/0 for all columns
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].map({True: 1, False: 0})
# Summary of Data
df_summary = df.describe()

"""

######################################

PREPROCESSING DATA
1.) Remove Near Zero Variance
2.) Remove Highly Correlated Variables
3.) Shape Distribution
    3A.) Transform skewness
    3B.) Yeo Johnson Transformation Formula
4.) Center and Scale to produce a standard normize distribution
5.) Impute/Model missing values
6.) Remove Outliers
    6A.) Can cause problems for some models like Principal Components Analysis & Linear/Logistic Regression
    6B.) Models resistant to outliers: Tree Models and Support Vector Machine
    6C.) Only remove outlier observations if there is a strong reason to do so
7.) Convert factor variables (string or categorical) to dummy variable

######################################

"""


"""

###################################################################
REMOVE NEAR ZERO VARIANCE
###################################################################


"""



def nearZeroVar(x=None, freqCut=95/5, uniqueCut=10, saveMetrics=False, names=False):
    '''This function replicates the nearZeroVar function in the R caret package.
    
     x          a pandas series or dataframe with all numeric data 
     freqCut    the cutoff for the ratio of the most common value to the second most common value
     uniqueCut  the cutoff percentage of distinct values out of the number of total samples
     names      a logical. If false, column indexes ar returned. If true, column names are returned.
    
     If saveMetrics=True then outputs
     freqRatio     the ratio of frequencies for the most common value over the second most common value
     percentUnique the percentage of unique data points out of the total number of data points
     zeroVar       a vector of logicals for whether the predictor has only one distinct value
     nzv           a vector of logicals for whether the predictor is a near zero variance predictor
     
     
     *** Code Enhancement: Includes handling for missing values, uses more efficient operations
                           and simplifies the logical checks.
                           
        Flexibility: This function can be easily modified to handle different types of data or incorporate
                    additional metrics. ***
     
     
     
     
     '''

    import numpy as np
    import pandas as pd
    
    results = []
    
    for col in x.columns:
        series = x[col].dropna()  # Handle missing values appropriately
        freq = series.value_counts()
        freqRatio = freq.iloc[0] / freq.iloc[1] if len(freq) > 1 else float('inf')
        percentUnique = 100 * len(pd.unique(series)) / len(series)
        zeroVar = len(freq) == 1

        if saveMetrics:
            results.append({
                'column': col,
                'freqRatio': freqRatio,
                'percentUnique': percentUnique,
                'zeroVar': zeroVar,
                'nzv': zeroVar or (freqRatio > freqCut and percentUnique <= uniqueCut)
            })
        else:
            if zeroVar or (freqRatio > freqCut and percentUnique <= uniqueCut):
                results.append(col)

    if saveMetrics:
        result_df = pd.DataFrame(results)
        return result_df.set_index('column') if not names else result_df
    else:
        return results

# Remove near zero variance predictors.
# Use the nearZeroVar function to find badColumns,
# i.e., those with near-zero variance.
indVarNames = [i for i in df if i not in 'price']
badNames = nearZeroVar(x=df.loc[:,indVarNames], names=True)

# print variables with near-zero variance.
print('Variables with near-zero variance in data: {}'.format(badNames))
# Remove the variables with near-zero variance.
df2 = df.loc[:,[i for i in df.columns if i not in badNames]]

# Confirm that variables in badColumns were dropped.
check = np.isin(df2.columns, badNames)

if sum(check)>0:
    print("{} in data2".format(df2.columns[check].tolist()))
else:
    print("bad columns in data were successfully removed")


"""
#################################################################################################
###  Remove Highly Correlated Variables ###
#################################################################################################

"""
def findCorrelation(x, cutoff = 0.9, verbose = False, names=False):
    '''This function replicates the findCorrelation function in the R caret package.
    
     x          a correlation matrix in the form of a pandas dataframe
     cutoff     a numeric value for the pair-wise absolute correlation cutoff
     verbose    a boolean for printing details
     names      a logical. If false, column indexes ar returned. If true, column names are returned.
    
    Returns a vector of indices denoting the columns to remove (when names=False), otherwise
    a vector of column names. If no correlations meet the criteria, then an empty list [].'''
    
    import numpy as np
    import pandas as pd
    
    if not isinstance(x, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if x.shape[0] != x.shape[1]:
        raise ValueError("Correlation matrix must be square.")

    # Use absolute values and mask the diagonal to avoid self-comparison
    x_abs = np.abs(x).to_numpy()
    np.fill_diagonal(x_abs, np.nan)
    
    to_delete = set()
    n = x.shape[1]

    # Sorting indices by mean correlation for each variable
    sorted_indices = np.argsort(np.nanmean(x_abs, axis=0))[::-1]

    for i in sorted_indices:
        if i in to_delete:
            continue
        for j in sorted_indices:
            if i != j and j not in to_delete and x_abs[i, j] > cutoff:
                mean_i = np.nanmean(x_abs[i])
                mean_j = np.nanmean(x_abs[j])
                if verbose:
                    print(f"Comparing {x.columns[i]} and {x.columns[j]} with correlation {x_abs[i,j]}")
                    print(f"Means: {mean_i} vs {mean_j}")
                # Decide which index to remove
                if mean_i > mean_j:
                    to_delete.add(i)
                    if verbose:
                        print(f"Removing {x.columns[i]}")
                    break
                else:
                    to_delete.add(j)
                    if verbose:
                        print(f"Removing {x.columns[j]}")

    # Return results based on the 'names' flag
    if names:
        return [x.columns[i] for i in to_delete]
    else:
        return list(to_delete)
    
    
# Remove highly correlated variables.

indVarNames = [i for i in df2 if i not in 'price']
badNames = findCorrelation(df2.loc[:,indVarNames].corr(), names=True)
print('Highly correlated variables in data to be removed:\n{}'.format(badNames))

df2 = df2.loc[:,[i for i in df2.columns if i not in badNames]]

# Confirm that variables in badColumns were dropped.
check = np.isin(df2.columns, badNames)

if sum(check)>0:
    print("{} in data2".format(df2.columns[check].tolist()))
else:
    print("bad columns in data2 were successfully removed")

# Look at the relationship between each predictor variable visually



def plotDepvarVsX(data, depVarCol, jit=False, lin=False, file=None):
    '''
    Plots the relationship between a dependent variable and all other variables.

    Parameters:
        data (DataFrame): A dataframe with all numeric data.
        depVarCol (int): The column index number of the dependent variable.
        jit (bool): If True, adds jitter to the plots.
        lin (bool): If True, draws a regression line on the plots.
        file (str): Path to save the PDF file with all plots.

    Returns:
        None. Plots are shown or saved to a file.
    '''
    
    if file is not None:
        pp = PdfPages(file)

    # Separate the dependent variable and predictors
    depVar = data.iloc[:, depVarCol]
    plotData = data.drop(columns=data.columns[depVarCol])
    
    # Iterate over each predictor to plot against the dependent variable
    for col in plotData.columns:
        plt.figure(figsize=(8, 4))
        if plotData[col].dtype == 'object' or len(plotData[col].unique()) <= 10:
            sns.boxplot(x=plotData[col], y=depVar)
        else:
            sns.scatterplot(x=plotData[col], y=depVar, alpha=0.6)
            if lin:
                sns.regplot(x=plotData[col], y=depVar, scatter=False, color='blue')
        
        plt.title(f'Relationship between {col} and {data.columns[depVarCol]}')
        plt.xticks(rotation=45, ha='right')

        if jit:
            sns.stripplot(x=plotData[col], y=depVar, color='red', jitter=0.1, size=3)

        # Save or display the plot
        if file is not None:
            pp.savefig(bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

    if file is not None:
        pp.close()

column_index = df2.columns.get_loc('price')
plotDepvarVsX(data=df2, depVarCol=column_index, jit=True, lin=True)

"""
##############################################################################
SHAPE DISTRIBUTION
##############################################################################
"""
# Check is variables are skewed
for i in df2:
    df2.loc[:,i].plot(kind='hist', title=i)
    plt.show()

# Check skewness of the variables.
df2.skew()

# Apply YeoJohnson to reduce skewness, and center and scale the data.
pt = PowerTransformer(method='yeo-johnson', standardize=True)

indVarNames = [i for i in df2 if i not in 'price']
df3 = pd.DataFrame(pt.fit_transform(df2), columns=df2.columns)

# Restore dependent variable to untransformed, 
# since we generally never want to transform the dependent variable.
df3.loc[:,'price'] = df2.loc[:,'price']

# Print the mean and std of untransformed and transformed features

check = pd.DataFrame({'untransf': df2.mean(), 'transf': df3.mean()})
print("\nMean of Untransformed and Transformed Features:\n{}" .format(check)) 

check = pd.DataFrame({'untransf': df2.std(), 'transf': df3.std()})
print("\nStd of Untransformed and Transformed Features:\n{}" .format(check)) 

# Print the skewness of untransformed and transformed features

check = pd.DataFrame({'untransf':df2.skew(), 'transf':df3.skew()})
print("\nskewness statistic of Untransformed and Transformed Features:\n{}" .format(check)) 

# Show Histograms for the orginal and transformed variables
for i in df3:    
    fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)
    fig.supxlabel(i)

    df2.loc[:,i].plot(kind='hist', ax=ax1)
    df3.loc[:,i].plot(kind='hist', ax=ax2)
    plt.show()

"""
##############################################################################
IMPUTE MISSING VALUES
##############################################################################
"""
look = df3.dropna(subset=['price'])  # Assume 'price' is fully observed and not imputed
features = look.drop(columns=['price'])

# Impute using different values of k and compare results
k_values = [1, 2, 3, 4, 30]
original_means = features.mean()
original_vars = features.var()

for k in k_values:
    imputer = KNNImputer(n_neighbors=k)
    features_imputed = imputer.fit_transform(features)
    features_imputed_df = pd.DataFrame(features_imputed, columns=features.columns)

    # Calculate means and variances post-imputation
    imputed_means = features_imputed_df.mean()
    imputed_vars = features_imputed_df.var()

    # Plotting for visual comparison
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.bar(features.columns, original_means, alpha=0.5, label='Original Means')
    plt.bar(features.columns, imputed_means, alpha=0.5, label=f'Imputed Means k={k}', color='r')
    plt.title('Mean Comparison')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.bar(features.columns, original_vars, alpha=0.5, label='Original Variances')
    plt.bar(features.columns, imputed_vars, alpha=0.5, label=f'Imputed Variances k={k}', color='r')
    plt.title('Variance Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
# The visual shows that k = 2 is the best 
imputer = KNNImputer(n_neighbors=2)

# Do the k-nearest neighbor imputation of missing values.
# Exclude only the 'price' column correctly.
indVarNames = [i for i in df3.columns if i.lower() != 'price']
df4 = imputer.fit_transform(df3[indVarNames])
df4 = pd.DataFrame(df4, columns=indVarNames)

# Check pre- and post-imputation values for 'monthlyHoaFee'.
# We create a copy to avoid any SettingWithCopyWarning
check = df3[['monthlyHoaFee']].copy()

# Now add the imputed 'monthlyHoaFee' values to the 'check' DataFrame
check['monthlyHoaFee_imputed'] = df4['monthlyHoaFee']

# Compare the mean before and after imputation for 'monthlyHoaFee'.
mean_before = check['monthlyHoaFee'].mean()
mean_after = check['monthlyHoaFee_imputed'].mean()

print(check.head())  # Show some of the changes
print(f'Mean before imputation for monthlyHoaFee: {mean_before}')
print(f'Mean after imputation for monthlyHoaFee: {mean_after}')

df4['price'] = df3['price'].values

"""
##############################################################################
BUILD LINEAR REGRESSION MODEL & LASSO REGRESSION MODEL
##############################################################################
"""

# Common setup for both models
predictors = df4.drop(columns=['price'])
correlations = predictors.corrwith(df4['price']).abs()
moderate_corr_features = correlations[correlations > 0.1].index.tolist()
reduced_predictors = predictors[moderate_corr_features]

# Prepare data
X = reduced_predictors
y = df4['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model_final = LinearRegression()
model_final.fit(X_train, y_train)
y_pred = model_final.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Lasso Regression Model with LassoCV to find optimal alpha
lasso_cv = LassoCV(cv=10, random_state=42, max_iter=10000)
lasso_cv.fit(X_train, y_train)
y_pred_lasso_cv = lasso_cv.predict(X_test)
r2_lasso_cv = r2_score(y_test, y_pred_lasso_cv)
rmse_lasso_cv = np.sqrt(mean_squared_error(y_test, y_pred_lasso_cv))

# Plotting both predictions on the same graph
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Linear Regression')
plt.scatter(y_test, y_pred_lasso_cv, alpha=0.5, color='red', label='Lasso Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Comparison of Actual vs. Predicted House Prices')
plt.legend()
plt.grid(True)
plt.show()

# Print the metrics
print("Linear Regression R²:", r2)
print("Linear Regression RMSE:", rmse)
print("Lasso Regression R²:", r2_lasso_cv)
print("Lasso Regression RMSE:", rmse_lasso_cv)

"""
##############################################################################
# TREE MODELS: Random Forest & XGBoost
##############################################################################
"""

# Assuming df4 is loaded and preprocessed
X = df4.drop(columns=['price'])
y = df4['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Initialize XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=100)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Plotting predictions of all models
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='green', label='Random Forest')
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color='purple', label='XGBoost')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Comparison of Actual vs. Predicted House Prices')
plt.legend()
plt.grid(True)
plt.show()

# Calculate metrics
print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("XGBoost R²:", r2_score(y_test, y_pred_xgb))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))

"""
##############################################################################
# OPTIMIZE TREE MODELS: Random Forest & XGBoost
##############################################################################
"""

# Random Forest Optimization
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search_rf = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
best_rf.fit(X_train, y_train)  # Refit with the best parameters
y_pred_rf = best_rf.predict(X_test)

# XGBoost Optimization
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'colsample_bytree': [0.3, 0.7]
}
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_
best_xgb.fit(X_train, y_train)  # Refit with the best parameters
y_pred_xgb = best_xgb.predict(X_test)

# Calculate and print optimized metrics
print("Optimized Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Optimized Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Optimized XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("Optimized XGBoost R²:", r2_score(y_test, y_pred_xgb))

"""
##############################################################################
# Imports and Data Preparation
##############################################################################
"""

# Assuming df4 is loaded and preprocessed
X = df4.drop(columns=['price'])
y = df4['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
##############################################################################
# Initialize and Fit Models
##############################################################################
"""

# Initialize Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Initialize XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=100)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

"""
##############################################################################
# Function to Plot Learning Curves
##############################################################################
"""

def plot_learning_curves(model, X_train, y_train, X_val, y_val):
    train_errors, val_errors = [], []
    for m in range(1, len(X_train), 50):  # Adjust step for large datasets
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.title("Learning Curves")
    plt.show()

"""
##############################################################################
# Plot Learning Curves
##############################################################################
"""

plot_learning_curves(rf, X_train, y_train, X_test, y_test)
plot_learning_curves(xgb_model, X_train, y_train, X_test, y_test)

"""
##############################################################################
# Cross-Validation for Generalization Error
##############################################################################
"""

rf_cv_scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=5)
xgb_cv_scores = cross_val_score(xgb_model, X, y, scoring='neg_mean_squared_error', cv=5)

print("Random Forest Cross-Validated RMSE:", np.sqrt(-rf_cv_scores))
print("XGBoost Cross-Validated RMSE:", np.sqrt(-xgb_cv_scores))

"""
##############################################################################
# GridSearchCV for Parameter Tuning
##############################################################################
"""

# Define parameter grids
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'colsample_bytree': [0.3, 0.7]
}

# Perform grid search
grid_search_rf = GridSearchCV(rf, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)
grid_search_xgb = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)

"""
##############################################################################
# Evaluate Optimized Models
##############################################################################
"""

best_rf = grid_search_rf.best_estimator_
best_rf.fit(X_train, y_train)  # Refit with the best parameters
best_xgb = grid_search_xgb.best_estimator_
best_xgb.fit(X_train, y_train)  # Refit with the best parameters

y_pred_rf = best_rf.predict(X_test)
y_pred_xgb = best_xgb.predict(X_test)

# Print optimized metrics
print("Optimized Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Optimized Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Optimized XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("Optimized XGBoost R²:", r2_score(y_test, y_pred_xgb))

# Actual vs Predicted plot
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='green', label='Random Forest Predictions')
plt.scatter(y_test, y_pred_xgb, alpha=0.6, color='blue', label='XGBoost Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Line')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Comparison of Actual vs. Predicted Prices')
plt.legend()
plt.grid(True)
plt.show()

# Calculate metrics
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_r2 = r2_score(y_test, y_pred_xgb)

# Print optimized metrics
print("Optimized Random Forest RMSE:", rf_rmse)
print("Optimized Random Forest R²:", rf_r2)
print("Optimized XGBoost RMSE:", xgb_rmse)
print("Optimized XGBoost R²:", xgb_r2)


# Assuming best_rf and best_xgb have been fitted to the training data
y_train_pred_rf = best_rf.predict(X_train)
y_train_pred_xgb = best_xgb.predict(X_train)

train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
train_r2_rf = r2_score(y_train, y_train_pred_rf)

train_rmse_xgb = np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))
train_r2_xgb = r2_score(y_train, y_train_pred_xgb)

print("Random Forest - Train RMSE:", train_rmse_rf, "Train R²:", train_r2_rf)
print("XGBoost - Train RMSE:", train_rmse_xgb, "Train R²:", train_r2_xgb)
