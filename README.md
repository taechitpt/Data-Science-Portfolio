# Data-Science-Portfolio
# 🧠 Taechit's Data Science Portfolio
👋 Hello! My name's **Taechit Khathanyaakemongkol**, Nickname: **Petong** 

---
## 🎓 Education
*M.Sc. in Data Science*, *Thammasat University (Expected to graduate : 2026)*  


---
## 🧰 Skills

| Category | Tools / Skills |
|-----------|----------------|
| **Programming** | Python, SQL, R|
| **Libraries** | Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, ggplot2 |
| **ML / AI** | Regression, Classification, Clustering, Decision Tree, Neural Networks, Association Rule Mining, NLP(Text Analytics)|
| **Visualization** | Excel, Power BI|
| **Tools** | RapidMiner |

---

## 🚀Academic Projects

### 📊 1. Risk factors contributing to heart attack                     
**Tools:** Python (Pandas, Matplotlib, sckit-learn).

**Goal:** Evaluation of a logistic regression model for predicting heart attack factors using kaggle dataset. 

**Process:**
- Developed and evaluated a logistic regression model to identify key risk factors associated with heart attacks using a public healthcare dataset.
- Achieved strong predictive performance through feature selection and model evaluation, supporting data-driven health insights.
  
**Result:** 
- In this case, the logistic regression model achieved an accuracy of 82%.
- The risk of heart attack varies according to age, male sex, diastolic blood pressure, and heart rate.


🔗 [Python code]
```python
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(file_path)

df['Result'] = df['Result'].map({'negative':0,'positive':1})
features = ['Age','Gender','Heart rate','Systolic blood pressure','Diastolic blood pressure','Blood sugar']
X = df[['Age','Gender','Heart rate','Systolic blood pressure','Diastolic blood pressure','Blood sugar']]
y = df['Result']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

coefficients = pd.Series(model.coef_[0], index=features)
print("Feature Coefficients:")
print(coefficients.sort_values)
#Visualization
plt.barh(features,coefficients)
plt.title("Impact of Features on Heart Attack Risk")
plt.xlabel('Coefficients values')
plt.ylabel('Features')
plt.axvline(x=0, color='r',linestyle='--')
plt.tight_layout()
plt.show()
```

### 💬 2. Trip advisor hotel reviews
**Tools:** RapidMiner (Text Analytics, Classification, Clustering).

**Goal:** Sentiment analysis of hotel reviews in relation to rating scores using the TripAdvisor dataset from kaggle.

**Process:**
- Conducted sentiment analysis on hotel reviews to examine the relationship between customer sentiment and rating scores.
- Built and evaluated supervised learning models (k-NN, Naive Bayes) and performed customer segmentation using clustering techniques.
- Delivered actionable insights through data visualization and interpretation of sentiment-driven patterns.

  
**Result:** 
- For the clustering phase, the k-Means algorithm with k = 3 yielded the lowest Davies-Bouldin index, successfully segmenting the data into three clusters. Cluster 1 mainly represents descriptive features of the hotel, while Clusters 2 and 3 represent positive and negative customer opinions, respectively.

🔗 [Trip Advisor Hotel Reviews - REPORT](https://drive.google.com/file/d/1YeklW6qZqACHYq9q5ZXrlH1o_ljV9pUd/view?usp=drive_link)

---



## 🏅 Certificates
- Python for Data Science, AI & Development – Coursera [ดูใบประกาศนียบัตร](https://drive.google.com/file/d/1yhW5Wkf7ViSJVGQo0GNJ6dJsH8SdpOpj/view?usp=drive_link)  
- Databases and SQL for Data Science with Python – Coursera [ดูใบประกาศนียบัตร](https://drive.google.com/file/d/1jVKPR2HJwHCzeegaDe3-YvgrH6gORjJu/view?usp=drive_link)  
- Machine Learning, Data Science & AI Engineering with Python – Udemy [ดูใบประกาศนียบัตร](https://drive.google.com/file/d/1b7TU7OlG_dOS3SPk1VN1YDhMpv3zXq1Z/view?usp=drive_link)   

---

## 📫 Contact
📧 Email: [techit.kha@gmail.com]  
🔗 GitHub: [[https://github.com/taechitpt](https://github.com/taechitpt)]  
🔗 LinkedIn: [[taechit-kh](https://www.linkedin.com/in/taechit-khathanyaakemongkol-2061a5337/)]  

---

> ✨ *“Data is not just numbers — it's a story waiting to be told.”*  
> — TaechitKh
