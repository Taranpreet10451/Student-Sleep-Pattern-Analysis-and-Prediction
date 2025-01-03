# Student-Sleep-Pattern-Analysis-and-Prediction

## Introduction
This project analyzes student sleep patterns and predicts sleep quality using various machine learning models. The dataset includes features like age, gender, university year, caffeine intake, screen time, physical activity, and study hours, which are used to predict the Sleep_Quality of students.

## Features

- Data Preprocessing: Handles missing values, cleans categorical data, and scales features.
- Data Visualization: Includes visualizations to explore relationships between variables and target labels.
- Class Imbalance Handling: Uses SMOTE to balance the dataset for better model training.

## Machine Learning Models: 

Trains and evaluates multiple models, including:
1. RandomForestClassifier
2. DecisionTreeClassifier
3. AdaBoostClassifier
4. GaussianNB
5. SVC
6. KNeighborsClassifier
7. XGBClassifier (XGBoost)
8. LGBMClassifier (LightGBM)
   
- Hyperparameter Tuning: Optimizes the XGBoost model using GridSearchCV.
- Performance Metrics: Evaluates models using accuracy, precision, recall, and F1-score.

## Installation

- Clone the repository:
git clone https://github.com/your-username/student-sleep-pattern.git
cd student-sleep-pattern

- Install the required dependencies:
pip install -r requirements.txt

## Requirements:

The following Python libraries are required:

1. matplotlib
2. seaborn
3. scikit-learn
4. imblearn
5. xgboost
6. lightgbm

## Usage:

- Prepare the Dataset: Place the dataset file (student_sleep_patterns.csv) in the specified path.
- Run the Script: Execute the script to preprocess the data, train models, and evaluate their performance.
  bash
  python sample.py

## Visualizations
- Line plots to observe trends between features and sleep quality.
- Count plots and pie charts for categorical data.
- Correlation heatmap for numerical features.
- Pairplots for exploring relationships between features.

## Models and Results

The following machine learning models are used, and their performance is compared:

| Model                  | Accuracy  | Notes                          |
|------------------------|-----------|--------------------------------|
| RandomForestClassifier | 27%       | Balanced performance overall. |
| DecisionTreeClassifier | 27%       | High interpretability.         |
| AdaBoostClassifier     | 12%       | Struggled with class imbalance.|
| GaussianNB             | 8%        | Weak on numerical features.    |
| SVC                    | 11%       | Sensitive to scaling.          |
| KNeighborsClassifier   | 19%       | Better recall for some classes.|
| XGBoostClassifier      | **30%**   | Best model after tuning.       |
| LightGBMClassifier     | 28%       | Comparable to XGBoost.         |


## Best Model

The XGBoost Classifier performed the best with:
- Accuracy: 30%
- Best Parameters:
  - learning_rate: 0.2
  - max_depth: 7
  - n_estimators: 100

## Recommendations for Improvement
- Address Class Imbalance: Improve handling of class imbalance using ensemble methods or more advanced resampling techniques.
- Feature Engineering: Add or combine features to better capture relationships.
- Model Ensemble: Use voting or stacking ensembles for better generalization.
- Data Augmentation: Generate synthetic samples to enrich the dataset.
