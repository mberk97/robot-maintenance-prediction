import sqlite3
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedKFold

# Connect to SQLite database
conn = sqlite3.connect('agv_db.db')
cursor = conn.cursor()

# Fetch all data, excluding potential data leakage columns with SQL
query = """
    SELECT "Type", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear", "Failure Type"
    FROM Test_table
    WHERE "Failure Type" IS NOT NULL;
"""
cursor.execute(query)
data = pd.DataFrame(cursor.fetchall(), columns=['Type', 'Air_Temperature', 'Process_Temperature', 'Rot_Speed', 'Torque', 'Tool_Wear', 'Failure_Type'])

# Close the cursor and connection
cursor.close()
conn.close()

# Separate 'No Failure' data because we don't need failured scenarios
no_failure_data = data[data['Failure_Type'] == 'No Failure']

# Filter only failure-related data for showing effects of cases
failure_data = data[data['Failure_Type'] != 'No Failure'].copy()
plt.figure(figsize=(10,5))
sns.countplot(data[data['Failure_Type'] != 'No Failure'],x="Failure_Type",hue='Failure_Type')
plt.title('Failure Types and Counts')
plt.show()

#Encoding Failure Types : No Failure ,Heat Dissipation Failure ,Power Failure ,Overstrain Failure ,Tool Wear Failure ,Random Failures    
encoder = LabelEncoder()
failure_data['Encoded_Type'] = encoder.fit_transform(failure_data['Type'])

print(failure_data)


# Define features and target variable to fit ML algorithms
X = failure_data[['Encoded_Type', 'Air_Temperature', 'Process_Temperature', 'Rot_Speed', 'Torque', 'Tool_Wear']]
y = encoder.fit_transform(failure_data['Failure_Type'])

# summary_stats help us to plot optimal values
# we can see mean,std,min,max values that we need in boxplot paramaters
summary_stats = no_failure_data[['Air_Temperature', 'Process_Temperature', 'Rot_Speed', 'Torque', 'Tool_Wear']].describe()


# Standardize features to improve convergence and prevent features with larger scales from dominating smaller ones.
scaler = StandardScaler()
X = scaler.fit_transform(X)
class_counts = np.bincount(y)

#Split data
test_size=0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


# cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ML Models : RANDOM FOREST- ENSEMBLE METHODS(GRADIENT BOOSTING,ADABOOST,BAGGING),Logistic Regression
models = {
    "Random Forest": (RandomForestClassifier(class_weight='balanced',random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [5, 10]
    }),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'learning_rate': [0.001, 0.01],
        'max_depth': [3, 5]
    }),
    "AdaBoost": (AdaBoostClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'learning_rate': [0.001, 0.01]
    }),
    "Bagging": (BaggingClassifier(random_state=42), {
        'n_estimators': [10, 50]
    }),
    "Logistic Regression": (LogisticRegression(class_weight='balanced',max_iter=5000), {})
}

best_model, best_f1 = None, 0
# Train and evaluate models
for model_name, (model, param_grid) in models.items():
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, error_score='raise')
    try:
        start_time = time.time()

        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)

        runtime = time.time() - start_time

        print(f"{model_name} Best Parameters: {grid_search.best_params_}, F1 Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_model, best_f1, best_model_name = best_estimator, f1, model_name
            best_accuracy, best_runtime, best_y_pred = accuracy, runtime, y_pred
    except ValueError as e:
        print(f"Skipping {model_name} due to error: {e}")

# Display results
if best_model:
    print(f"BestModel is: {best_model_name}")
    print(f"Accuracy of model: {best_accuracy * 100:.1f}%")
    print(f"Training Runtime in seconds: {best_runtime:.3f}")
    print("Classification Matrix:\n")
    print(classification_report(y_test, best_y_pred, digits=2))

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.show()
else:
    print("No suitable model found.")

# Plot optimal parameter ranges that we obtained from summary_stats per product type 
plt.figure(figsize=(10, 6))
sns.boxplot(x='Type', y='Air_Temperature',hue="Type",data=no_failure_data)
plt.xticks(rotation=45)
plt.xlabel("Component Type")
plt.ylabel("Air Temperature (°K)")
plt.title("Optimal Air Temperature Range per Component " )
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Type', y='Process_Temperature',hue="Type",data=no_failure_data)
plt.xticks(rotation=45)
plt.xlabel("Component Type")
plt.ylabel("Process Temperature (°K)")
plt.title("Optimal Process Temperature Range per Component ")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Type', y='Rot_Speed',hue="Type", data=no_failure_data)
plt.xticks(rotation=45)
plt.xlabel("Component Type")
plt.ylabel("Rotational Speed(rpm)")
plt.title("Optimal Rotational Speed Range per Component ")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Type', y='Torque',hue="Type", data=no_failure_data)
plt.xticks(rotation=45)
plt.xlabel("Component Type")
plt.ylabel("Torque (Nm)")
plt.title("Optimal Torque Range per Component ")
plt.show()
