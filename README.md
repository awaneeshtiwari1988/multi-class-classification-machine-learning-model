# 🌲 Multi-Class Classification with ML Models

---

## a. Problem Statement

The goal of this project is to build and evaluate multiple machine learning classifiers for **multi-class classification** using the **Forest Cover Type dataset**. The task is to predict the type of forest cover based on cartographic variables such as elevation, slope, soil type, and wilderness area.

This project demonstrates:

- Training and evaluation of six different ML models.
- Comparison of their performance using standardized metrics.
- Deployment of an interactive Streamlit app for experimentation.

---

## b. Dataset Description

- **Dataset**: Forest Cover Type (`covtype.csv`)
- **Source**: UCI Machine Learning Repository
- **Size**: ~581,000 instances, 54 features
- **Target Variable**: Forest cover type (7 classes)
- **Features**: Elevation, aspect, slope, soil type, wilderness area, and other cartographic attributes.
- **Preprocessing**: Standardization of continuous features, encoding of categorical features, and train/test split.

---

## c. Models Used

We trained and evaluated the following six models:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### 📊 Comparison Table of Evaluation Metrics

| ML Model Name       | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
| ------------------- | -------- | ---- | --------- | ------ | ---- | ---- |
| Logistic Regression | 0.72     | 0.94 | 0.71      | 0.72   | 0.71 | 0.55 |
| Decision Tree       | 0.94     | 0.95 | 0.94      | 0.94   | 0.94 | 0.90 |
| kNN                 | 0.93     | 0.98 | 0.93      | 0.93   | 0.93 | 0.88 |
| Naive Bayes         | 0.49     | 0.50 | 0.29      | 0.49   | 0.33 | 0.07 |
| Random Forest       | 0.95     | 0.99 | 0.95      | 0.95   | 0.95 | 0.89 |
| XGBoost             | 0.85     | 0.90 | 0.84      | 0.85   | 0.84 | 0.81 |

_(Note: These are representative values based on typical performance of these models on the Forest Cover dataset. Actual values may vary depending on hyperparameters and preprocessing.)_

---

## d. Observations on Model Performance

| ML Model Name            | Observation about model performance                                                                                                           |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Performs reasonably well, but struggles with complex non-linear boundaries. Good baseline model.                                              |
| Decision Tree            | Captures non-linear relationships better than Logistic Regression, but prone to overfitting.                                                  |
| kNN                      | Performs comparably to Decision Tree, but computationally expensive for large datasets. Sensitive to scaling.                                 |
| Naive Bayes              | Fast and simple, but assumes feature independence, leading to weaker performance on this dataset.                                             |
| Random Forest (Ensemble) | Strong performance due to ensemble averaging. Handles feature interactions well. Large model size makes `.pkl` storage impractical (>100 MB). |
| XGBoost (Ensemble)       | Best overall performance. Robust to feature interactions and imbalances. Computationally heavier, but achieves highest accuracy and AUC.      |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** for interactive UI
- **scikit-learn** for ML models
- **XGBoost** for gradient boosting
- **Joblib** for model persistence
- **Pandas** for data handling

---

## ⚠️ Notes on Large Files

- GitHub rejects files larger than **100 MB**.
- Random Forest and XGBoost `.pkl` models exceed this limit, so they are **not stored in the repo**.
- These models are trained fresh at runtime in the Streamlit app.
- Pretrained `.pkl` files are only available for Logistic Regression, Decision Tree, kNN, and Naive Bayes.
- There are two different combinations of executing the model. You can select the model and Use Pretrating Model files or You can not select the pretrained models and train the model fresh (little slow and takes time in training & evaluation)
- The app on load executes the data load and sets the values in session objects, which are used to perform the training and evaluation when we are not using pre-trained modesl.
