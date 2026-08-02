# Wine Classification Benchmark: SVM vs Random Forest

**Author:** Joseph Eniola Kehinde  
**Context:** Applied Machine Learning Practicum – Foundation for Agricultural Sensor Classification.

##  Overview
This project implements and compares **Support Vector Machine (SVM)** and **Random Forest (RF)** classifiers on the UCI Wine dataset. 
Unlike a basic tutorial, this implementation includes:
- **Hyperparameter Tuning** (GridSearchCV)
- **Cross-Validation** (5-fold)
- **Feature Importance Analysis** (for interpretability)
- **Noise Robustness Testing** (simulating real-world sensor degradation)

##  Key Findings
- Tuned SVM achieved **~98%** accuracy with RBF kernel (C=10, gamma=0.01).
- Random Forest showed slightly lower accuracy but provided **interpretable feature importances** (Proline and Flavanoids are top predictors).
- Both models proved robust via cross-validation (std < 0.02).

## Extension to Original Research
This workflow serves as the **baseline pipeline** for my proposed PhD research on **Trustworthy AI for Livestock Monitoring**. 
Just as we scale chemical features here, I will scale 3D skeletal keypoints (from dairy sheep/pork) using similar preprocessing and classification frameworks, 
with an added **Uncertainty Quantification** layer.

## Results
Confusion matrices and feature importance plots are saved in `/results`.

##  How to Run
1. Clone the repo or download the files.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python wine_analysis.py`
4. Run robustness test: `python robustness_test.py`
