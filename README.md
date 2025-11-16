# Heart Attack Risk Prediction – Machine Learning Project

This project builds a complete end-to-end **Heart Attack Risk Prediction** pipeline using machine learning.  
It involves data preprocessing, feature engineering, dimensionality reduction, training classification models,  
and evaluating performance using multiple metrics.

The dataset contains **8,763 samples** and covers demographic, lifestyle, medical measurements, and geographical features.  
The target variable is **Heart Attack Risk (0 or 1)**.

---

## 🚀 Project Features

### **1. Data Preprocessing**
- Loaded dataset from Google Drive (Colab environment).
- Cleaned and explored the dataset.
- One-hot encoded categorical variables:
  - `Sex`, `Diet`, `Country`, `Continent`,  
    `Blood Pressure`, `Hemisphere`
- Removed irrelevant identifiers such as:
  - `Patient ID`
- Combined encoded categorical features with numerical ones.
- Final dataset shape after encoding: **3966 features**

---

## 🔍 Exploratory Data Analysis (EDA)
The project includes:
- Pairplot visualizations (Seaborn)
- Correlation heatmap
- Scatterplots for linearity and correlation

---

## 🌲 Random Forest Feature Importance
To reduce dimensionality:
- Trained `RandomForestClassifier`
- Extracted feature importance for all 3966 features
- Selected all **non-zero importance features**  
  → Reduced feature set significantly while keeping meaningful predictors.

---

## 📉 Dimensionality Reduction (LDA)
- Applied **Linear Discriminant Analysis** (LDA)
- Reduced dataset into **one LDA component**  
  (since the dataset is binary classification)
- Visualized:
  - Training LDA distribution
  - Test LDA distribution
  - Combined visualization

---

## 🤖 Model Training
A `RandomForestClassifier` was trained on the LDA-transformed features.

### **Model Performance**
- **Accuracy:** ~0.537  
- **Confusion Matrix**
- **ROC Curve** with AUC
- **Precision-Recall Curve**
- **Threshold vs. F1 Curve**

---

## 📈 Additional Analysis
- Linear regression performed on sample custom data points
- Assessed linearity using:
  - R² value
  - Scatter plot
  - Pearson correlation coefficient

Conclusion: outcome was **non-linear**.

---

## 📁 Project Structure

├── data/
│ └── heart_attack_prediction_dataset.csv
├── notebooks/
│ └── heart_attack_analysis.ipynb
├── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ └── evaluation.py
├── images/
│ ├── heatmap.png
│ ├── lda_plot.png
│ └── roc_curve.png
├── README.md
└── requirements.txt

yaml
Copy code

*(You may adjust structure according to your folder layout.)*

---

## 🧪 Technologies Used

| Category | Tools |
|---------|-------|
| Language | Python |
| ML & Preprocessing | scikit-learn, imbalanced-learn |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab (+ Google Drive) |
| Data Handling | Pandas, NumPy |

---

## 📊 Results Summary

| Component | Result |
|----------|---------|
| Model | RandomForestClassifier |
| Dimensionality Reduction | LDA |
| Accuracy | **0.537** |
| ROC-AUC | Plotted |
| Feature Count Before | 3966 |
| Feature Count After RF Reduction | Significantly reduced |
| Data Size | 8,763 rows |

---

## 🔮 Future Improvements (Recommended Enhancements)

### ✔ **1. Replace LDA with PCA or UMAP**
LDA works only with linearly separable data.  
Your dataset is **high-dimensional, noisy, and nonlinear**, so PCA or UMAP will give better representation.

### ✔ **2. Feature Engineering for Blood Pressure**
Instead of using thousands of one-hot encoded values like `152/98`:
- Split into Systolic & Diastolic columns  
  → `BP_Systolic`, `BP_Diastolic`

This alone will reduce thousands of columns to **two** useful features.

### ✔ **3. Class Imbalance Handling**
Use:
- SMOTE  
- RandomUnderSampler  
- ADASYN  
- BalancedRandomForest

### ✔ **4. Try Advanced Algorithms**
- XGBoost  
- LightGBM  
- CatBoost (handles categorical features directly)

### ✔ **5. Hyperparameter Optimization**
Use:
- GridSearchCV  
- RandomizedSearchCV  
- Optuna (best option)

### ✔ **6. Model Explainability**
- SHAP (global and local interpretation)
- LIME

### ✔ **7. Deploy the Model**
- Build a Flask/FastAPI API
- Deploy on HuggingFace Spaces or Render
- Create an interactive dashboard using Streamlit

---

## 📜 License
This project is open-source under the **MIT License**.

---

## 🧑‍💻 Author
**Lajim**  
Machine Learning & Data Science Enthusiast  
(Feel free to modify this section)

---

## ⭐ Contributions
Pull requests are welcome!  
If you use this project, consider giving the repo a **star** ⭐
