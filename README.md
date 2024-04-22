# Zillow

Extracting Zillow housing data and running classical models for housing prediction

## Prerequisites

- Python 3.8+
- Rapid API key

## Installation

1. Clone the repository: `git clone https://github.com/your-repo`
2. Install dependencies: `pip install -r requirements.txt`

## Instructions
1.) Change Locations in Zillow_Test_File.py

## Usage

To gather the housing data, run the main script:
python Zillow_Test_File.py


Afterwards, to perform tests, run:
python Regression_Zillow.py
This will pull the csv output of the zillow details

## Regression Script will do the following:
1.) Preprocessing
  1A.) Get rid of nearZero Variance
  1B.) Get rid of highly correlated variables
  1C.) Shape Distribution
    - Transform Skewness
    - Yeo Johnson Transformation Formula
  1D.) Center and Scale to produce a standard normalize distribution
  1E.) Impute/Model missing values
  1F.) Remove Outliers
    Can cause problems for some modles like Principal Components Analysis & Linear/Logistic Regression
    - Models resistant to outliers: Tree Models and Suppor Vector Machine
    - Only remove outliers observations if there is a strong reason to do so
  1G.) Convert factor variables (string or categorical) to dummy variables (if needed, the script does this first)
2.) Linear & Lasso Regression Model
3.) Tree Models: Random Forest & XGBoost
4.) Check for overfitting
5.) Optimize Models
6.) Evaluate Optimize Models
## Features and Limitations

### Features:
- Feature 1
- Feature 2

### Limitations:
- The free version is limited to 100 requests.
- With a subscription, you can pull 10,000+ requests.


