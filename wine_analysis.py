# wine_analysis.py - Advanced Classification on Wine Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("WINE CLASSIFICATION BENCHMARK")
print("="*50)

# 1. Load Data
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
target_names = wine.target_names
print(f"Dataset shape: {X.shape}, Classes: {len(target_names)}")

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Feature Scaling (Crucial for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model 1: SVM with Hyperparameter Tuning (GridSearch)
print("\n--- Tuning SVM ---")
svm_param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
svm_grid = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)
best_svm = svm_grid.best_estimator_
print(f"Best SVM Params: {svm_grid.best_params_}")

# 5. Model 2: Random Forest with Hyperparameter Tuning
print("\n--- Tuning Random Forest ---")
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_
print(f"Best RF Params: {rf_grid.best_params_}")

# 6. Cross-Validation Scores (Robustness check)
svm_cv = cross_val_score(best_svm, X_train_scaled, y_train, cv=5)
rf_cv = cross_val_score(best_rf, X_train_scaled, y_train, cv=5)
print(f"\nSVM CV Accuracy: {svm_cv.mean():.4f} (+/- {svm_cv.std():.4f})")
print(f"RF CV Accuracy: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")

# 7. Evaluation Function
def evaluate(y_true, y_pred, model_name):
    print(f"\n--- {model_name} Test Set Results ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (weighted): {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall (weighted): {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    return confusion_matrix(y_true, y_pred)

# 8. Test Predictions & Results
y_pred_svm = best_svm.predict(X_test_scaled)
y_pred_rf = best_rf.predict(X_test_scaled)

cm_svm = evaluate(y_test, y_pred_svm, "Tuned SVM")
cm_rf = evaluate(y_test, y_pred_rf, "Tuned Random Forest")

# 9. Plot Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(cm_svm, display_labels=target_names).plot(ax=axes[0], cmap='Blues')
axes[0].set_title("Tuned SVM")
ConfusionMatrixDisplay(cm_rf, display_labels=target_names).plot(ax=axes[1], cmap='Greens')
axes[1].set_title("Tuned Random Forest")
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

# 10. Feature Importance (Random Forest)
importances = best_rf.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(10,6))
feat_imp.plot(kind='bar')
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("\n--- Feature Importances ---")
print(feat_imp)
