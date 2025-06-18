# Customer Web Analytics Clustering & Outlier Detection using ML

This project applies machine learning techniques to perform **clustering analysis**, **outlier detection**, and **classification modeling** on web analytics data. The dataset contains customer behavior metrics such as bounce rate, session duration, page views, etc. The goal is to uncover hidden patterns and improve decision-making using unsupervised and supervised ML models.

---

## 📊 Dataset

- File: `Boun_rate_2001.csv`
- Contains web traffic and customer interaction metrics:
  - Page Views
  - Session Duration
  - Bounce Rate
  - Time on Page
  - Previous Visits
  - Conversion Rate
  - Traffic Source

---

## ✅ Key Features

- **Exploratory Data Analysis (EDA):** Pie charts, histograms, scatter plots.
- **Outlier Detection:**
  - Z-score
  - Winsorization
  - Isolation Forest
- **Dimensionality Reduction:**
  - PCA (2D & 3D Visualizations)
- **Clustering Algorithms:**
  - K-Means
  - Gaussian Mixture Model (GMM)
  - DBSCAN
- **Classification Models (Post-Clustering):**
  - Random Forest
  - Logistic Regression
  - XGBoost
  - Support Vector Machine (SVM)
- **Visualizations:**
  - Cluster plots
  - Feature importance
  - Pair plots

---

## 🧠 Machine Learning Workflow

1. Load and clean the dataset
2. Encode categorical variables
3. Visualize feature distributions
4. Detect and treat outliers
5. Apply dimensionality reduction using PCA
6. Perform clustering:
   - Compare KMeans, GMM, DBSCAN
7. Detect and remove outliers using Isolation Forest
8. Retrain clustering models
9. Train classifiers to predict clusters
10. Evaluate accuracy and feature importance

---

## 📌 Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
