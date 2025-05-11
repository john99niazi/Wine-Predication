


# 🍷 Wine Quality Prediction - Good or Not?

This project uses **machine learning** to predict whether a white wine is **"good"** or **"not good"** based on its physicochemical properties.
The dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

---

## 📁 Dataset Overview

The dataset includes **12 features**:

* `fixed acidity`
* `volatile acidity`
* `citric acid`
* `residual sugar`
* `chlorides`
* `free sulfur dioxide`
* `total sulfur dioxide`
* `density`
* `pH`
* `sulphates`
* `alcohol`
* `quality` (score from 0 to 10)

---

## 🧪 Objective

The goal is to transform the `quality` column into a **binary classification problem**:

* ✅ Good Wine → `quality >= 7` → Label: **1**
* ❌ Not Good Wine → `quality < 7` → Label: **0**

Build a classification model to predict this label.

---

## ⚙️ Data Preprocessing

### 📌 1. Selected Features

Based on correlation analysis, we retained these five features for modeling:

```python
features = ['alcohol', 'density', 'volatile acidity', 'chlorides', 'total sulfur dioxide']
```

### 🏷️ 2. Label Conversion

The binary target label was created as follows:

```python
white['quality_label'] = (white['quality'] >= 7).astype(int)
```

### ⚠️ 3. Outlier Removal (Optional)

Used the IQR method to filter out outliers from numeric columns:

```python
outlier_indices = set()
for col in white.select_dtypes(include='number').columns:
    Q1 = white[col].quantile(0.25)
    Q3 = white[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_indices.update(white[(white[col] < lower) | (white[col] > upper)].index)

white_clean = white.drop(index=outlier_indices)
```

---

## 🤖 Model Training

### 📊 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X = white[features]
y = white['quality_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 🧠 Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## 📈 Model Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))
```

---

## 🔍 Predict from User Input

```python
alcohol = float(input("Enter alcohol: "))
density = float(input("Enter density: "))
volatile_acidity = float(input("Enter volatile acidity: "))
chlorides = float(input("Enter chlorides: "))
total_sulfur = float(input("Enter total sulfur dioxide: "))

sample = [[alcohol, density, volatile_acidity, chlorides, total_sulfur]]
result = model.predict(sample)

if result[0] == 1:
    print("✅ The wine is GOOD")
else:
    print("❌ The wine is NOT GOOD")
```

---

## 📊 Visualizations

Created with **Seaborn** and **Matplotlib**:

* 📦 **Boxplots** to detect outliers
* 🔥 **Correlation heatmap**
* 🧮 **Pairplots** for multivariate relationships
* 📉 **Scatter plots** for visual trends

---

## ✅ Key Findings

* **Alcohol** shows the strongest positive correlation with wine quality.
* Logistic Regression performed well as a baseline model.
* Accuracy was reasonable on both train and test sets.

---

## 🚀 Future Improvements

* Try **advanced models**: Random Forest, XGBoost
* Apply **hyperparameter tuning**
* Add a **Streamlit GUI** for user-friendly predictions
* **Deploy** the model using Flask or FastAPI

---

## 📚 References

* [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
* [Scikit-learn Documentation](https://scikit-learn.org/)
* [Seaborn Documentation](https://seaborn.pydata.org/)
* [Matplotlib Documentation](https://matplotlib.org/)

---


