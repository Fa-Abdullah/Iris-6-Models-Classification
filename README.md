# Iris 6 Models Classification

A comprehensive machine learning project on the Iris dataset, comparing 6 different classification models with hyperparameter tuning and neural network implementation.

## Dataset

- Source: Iris dataset (Kaggle)
- Samples: 150
- Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- Target: Species (3 classes: Iris-setosa, Iris-versicolor, Iris-virginica)

## Preprocessing

- Label Encoding for target variable
- Outlier Handling using IQR (Interquartile Range) method
- StandardScaler for feature normalization (applied before SVM)
- Data Splitting: 70% train, 30% test

## Exploratory Data Analysis

- Correlation heatmap
- Species distribution (balanced: 50 samples each)
- Box plots for outlier detection
- Feature distributions

## Models Compared (6)

### 1. Logistic Regression (Multinomial)
Accuracy: ~97.8%

### 2. SVM (Support Vector Machine)
- Hyperparameter Tuning: GridSearchCV (C, kernel)
- Best Params: C=1, kernel='rbf'
- Accuracy: ~97.8%

### 3. Decision Tree
- Hyperparameter Tuning: GridSearchCV (criterion, max_depth)
- Best Params: criterion='gini', max_depth=6
- Accuracy: ~97.8%
- Visualization: Tree plot included

### 4. Random Forest
- Hyperparameter Tuning: GridSearchCV (n_estimators, max_features, max_depth, criterion)
- Best Params: n_estimators=200, max_features='sqrt', max_depth=4, criterion='gini'
- Accuracy: ~97.8%

### 5. XGBoost
- Hyperparameter Tuning: GridSearchCV (min_child_weight, gamma, subsample, colsample_bytree, max_depth)
- Best Params: colsample_bytree=0.6, gamma=0.5, max_depth=3, min_child_weight=1, subsample=0.8
- Accuracy: ~97.8%

### 6. Neural Network (Keras)
- Architecture:
  - Input: 4 features
  - Hidden: Dense(100, ReLU)
  - Output: Dense(3, Softmax)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 45
- Test Accuracy: ~93.3%

## Performance Comparison

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 97.8% |
| SVM (RBF) | 97.8% |
| Decision Tree | 97.8% |
| Random Forest | 97.8% |
| XGBoost | 97.8% |
| Neural Network | 93.3% |

## Visualizations

- Confusion Matrix for each model
- Decision Tree plot
- Training/Validation curves for Neural Network
- Correlation heatmap
- Species distribution bar chart
- Box plots (before/after outlier handling)

## Author

Fatma Abdullah
