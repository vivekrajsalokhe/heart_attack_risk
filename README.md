# 🫀 Heart Attack Prediction using Random Forest

This project uses a **Random Forest Classifier** to predict the likelihood of a heart attack based on patient health indicators.  
It demonstrates a full end-to-end machine learning workflow — from data preprocessing to model evaluation — using Python and scikit-learn.

---

## 📘 Overview

Heart disease is one of the leading causes of death globally. Early prediction using data-driven models can help in preventive care and timely treatment.  
This notebook trains and evaluates a **Random Forest model** to classify whether a patient is at risk of a heart attack.

---

## 🧩 Project Workflow

### 1. Import Libraries
Essential Python libraries used:
- **pandas** for data manipulation  
- **numpy** for numerical operations  
- **matplotlib** and **seaborn** for visualization  
- **scikit-learn** for model building and evaluation  

### 2. Load Dataset
- Dataset used: `heart_attack_china.csv`  
- The notebook loads and explores the dataset using:
  ```python
  data.head(), data.info(), data.describe()
  ```
- Checks for missing values and data types.

### 3. Data Preprocessing
- Removes missing values using:
  ```python
  data = data.dropna()
  ```
- Converts categorical columns into numeric format with:
  ```python
  pd.get_dummies(data, drop_first=True)
  ```
- Identifies the **target column** (containing “heart” in its name).

### 4. Feature Selection
Splits data into features (**X**) and target (**y**):
```python
X = data.drop(columns=target_column)
y = data[target_column]
```

### 5. Train-Test Split
- Splits the data into 80% training and 20% testing sets.
- Uses a fixed random seed (`random_state=42`) for reproducibility.

### 6. Model Training
Trains a Random Forest Classifier:
```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

### 7. Prediction & Evaluation
Evaluates the model using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix**

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

## 📊 Example Outputs

| Metric | Description |
|:--|:--|
| **Accuracy** | Measures overall performance of the classifier |
| **Precision / Recall / F1-Score** | Evaluates classification quality |
| **Confusion Matrix** | Visualizes prediction vs actual outcomes |

---

## 🧠 Model Details

**Algorithm:** Random Forest Classifier  
**Key Features:**
- Ensemble method combining multiple decision trees  
- Handles both categorical and continuous variables  
- Reduces overfitting and improves generalization  

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install Dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Run the Notebook**
   ```bash
   jupyter notebook "random forest heart attack.ipynb"
   ```

4. **Provide Dataset**
   - Place your dataset file `heart_attack_china.csv` in the same directory as the notebook.

---

## 🚀 Future Improvements

- Use **GridSearchCV** for hyperparameter tuning  
- Visualize **feature importances**  
- Implement **missing value imputation** instead of dropping rows  
- Add **ROC Curve / AUC** metrics for better evaluation  
- Compare results with other models like **Logistic Regression**, **SVM**, or **XGBoost**

---

## 📈 Example Visualizations

The notebook includes:
- **Pairplots** and **correlation heatmaps** using Seaborn  
- **Feature importance bar plots** from the Random Forest model  

These visualizations help in understanding which features most influence heart attack risk.

---

## 🧾 File Structure

```
├── random forest heart attack.ipynb   # Jupyter Notebook
├── heart_attack_china.csv             # Dataset file (not included in repo)
└── README.md                          # Project documentation
```

---

## 🪪 License

This project is open-source and available under the **MIT License**.

---

## 👨‍💻 Author

**Vivek Salokhe**  
💼 Machine Learning Enthusiast | 💡 Data Science Learner  
📧 viveksalokhe1999@gmail.com

---

## ⭐ Acknowledgements

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)
- Dataset inspired by public heart disease datasets (e.g., Kaggle) [Link](https://www.kaggle.com/datasets/ankushpanday2/heart-attack-risk-dataset-of-china)

---

> 🩺 *“Prevention is better than cure — data-driven insights make prevention possible.”*
