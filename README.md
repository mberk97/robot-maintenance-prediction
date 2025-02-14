
# Robot Maintenance Prediction

This project aims to predict maintenance needs for robotic components by analyzing key operational parameters. It utilizes machine learning models to classify failure types and identify optimal working conditions.

## Dataset

### Dataset Source

The dataset used in this project is sourced from Kaggle:  
[Machine Failure Predictions Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/machine-failure-predictions)  

This dataset contains operational data from industrial machinery, including temperature, rotational speed, torque, and tool wear, along with failure labels.

The dataset includes operational metrics for robotic components:
- **UDI**
- **Product ID**
- **Type**
- **Air temperature [K]**
- **Process temperature [K]**
- **Rotational speed [rpm]**
- **Torque [Nm]**
- **Tool wear [min]**
- **Failure Type** (classification target)

Data is sourced from an SQLite database.

## Features

- **Data Preprocessing**: Encoding categorical variables, handling missing values, and feature scaling.
- **Failure Type Analysis**: Visualization of failure type distribution.
- **Machine Learning Models**:
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - Bagging
  - Logistic Regression
- **Hyperparameter Tuning**: GridSearchCV for optimal model selection.
- **Cross-Validation**: K-Fold cross-validation to ensure model robustness.
- **Performance Metrics**: Accuracy, F1-score, and confusion matrix for evaluation.
- **Optimal Condition Analysis**: Boxplots for temperature, speed, and torque per component type.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/mberk97/robot-maintenance-prediction.git
   cd robot-maintenance-prediction
