import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import tree,ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings 
warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv('C:/Users/Taran/Desktop/Student Sleep Pattern/student_sleep_patterns.csv')
print(data.head())
print(data.tail())
print(data.isnull().sum())
print(data.nunique())
print(data.describe())
print(data.info())
# Basic data preprocessing
col = ['Weekday_Sleep_Start', 'Weekday_Sleep_End', 'Weekend_Sleep_Start', 'Weekend_Sleep_End', 'Student_ID']
data = data.drop(col, axis=1)

# Clean and convert 'University_Year' column
data['University_Year'] = data['University_Year'].str.replace('st Year','')
data['University_Year'] = data['University_Year'].str.replace('nd Year','')
data['University_Year'] = data['University_Year'].str.replace('rd Year','')
data['University_Year'] = data['University_Year'].str.replace('th Year','')
data['University_Year'] = data['University_Year'].astype(int)

print(data['University_Year'].value_counts())
print(data.head())

# Data visualization
columns = ['Age', 'Gender', 'University_Year', 'Caffeine_Intake', 'Screen_Time', 'Physical_Activity', 'Sleep_Duration']
for col in columns:
    x = data.groupby([col])['Sleep_Quality'].mean().reset_index()
    sns.lineplot(x=col, y='Sleep_Quality', data=x)
    plt.title(f'Sleep Quality vs {col}')
    plt.show()

sns.countplot(data=data, x='Caffeine_Intake', hue='Gender')
plt.title('Gender vs Caffeine Intake')
plt.show()

data['University_Year'].value_counts().plot(kind='pie', stacked=True, autopct="%1.1f%%", legend=True)
plt.show()

data['Gender'].value_counts().plot(kind='pie', stacked=True, autopct='%1.1f%%', legend=True)
plt.show()

df = data.groupby("Study_Hours").agg({"Sleep_Quality":"mean"}).reset_index()
plt.figure(figsize=(35,10))
sns.barplot(data=df, x='Study_Hours', y='Sleep_Quality', legend=True)
plt.show()

sns.pairplot(data[['Study_Hours', 'Sleep_Quality']])
plt.show()

sns.pairplot(data[['Screen_Time', 'Sleep_Quality']])
plt.show()

# Encoding categorical variables
data['Gender'] = data['Gender'].str.replace('Male', '1')
data['Gender'] = data['Gender'].str.replace('Female', '2')
data['Gender'] = data['Gender'].str.replace('Other', '3')
data['Gender'] = data['Gender'].astype(int)
print(data['Gender'].value_counts())

print(data.head())
print(data.shape)
print(data.columns)
# Correlation heatmap
numeric_columns = list(data.select_dtypes(include=['int64', 'float64']))
print(numeric_columns)
sns.heatmap(data[numeric_columns].corr().abs(), annot=True)
plt.show()

# Split data into features and target
x = data.drop('Sleep_Quality', axis=1)
y = data['Sleep_Quality']

# Adjust target variable 'y' to have class labels starting from 0
y = y - 1  # Shifting the labels to start from 0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape)
print(y_train.shape)

# SMOTE for balancing classes
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Models with improvements
model1 = ensemble.RandomForestClassifier(random_state=42)
model1.fit(x_train, y_train)
model2 = tree.DecisionTreeClassifier(random_state=42)
model2.fit(x_train, y_train)
model3 = ensemble.AdaBoostClassifier(random_state=42)
model3.fit(x_train, y_train)
model4 = GaussianNB()
model4.fit(x_train, y_train)
model5 = SVC(probability=True, random_state=42)
model5.fit(x_train, y_train)
model6 = KNeighborsClassifier()
model6.fit(x_train, y_train)
model7 = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
model7.fit(x_train, y_train)  # Fixed: Removed -1 operation on x_train and y_train
model8 = LGBMClassifier(random_state=42)
model8.fit(x_train, y_train)

models = [model1, model2, model3, model4, model5, model6, model7, model8]
acc = []
titles = []

# Evaluate models
for model in models:
    pred = model.predict(x_test)
    model_acc = accuracy_score(y_test, pred)
    acc.append(model_acc)
    titles.append(type(model).__name__)
    print(f'{type(model).__name__} Classification Report:\n', classification_report(y_test, pred))
    print(f'{type(model).__name__} Accuracy: {model_acc}\n')

# Hyperparameter Tuning for the best model 
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_xgb = GridSearchCV(XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42),
                        param_grid, cv=5, scoring='accuracy')
grid_xgb.fit(x_train, y_train)  # Fixed: Removed -1 operation here too

print("Best Parameters for XGBoost:", grid_xgb.best_params_)
print("Best Cross-Validation Accuracy for XGBoost:", grid_xgb.best_score_)

# Plot accuracies
fig = plt.figure(figsize=(12, 6))
sns.barplot(x=titles, y=acc, palette='viridis')
plt.title('Model Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()