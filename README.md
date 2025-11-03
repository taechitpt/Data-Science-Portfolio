# Data-Science-Portfolio
# 🧠 Taechit's Data Science Portfolio
👋 Hello! My name's **Taechit Khathanyaakemongkol** Nickname: **Petong** 

---
## 🎓 Education
*M.Sc. in Data Science*, *Thammasat University (Expected to graduate : 2026)*  


---
## 🧰 Skills

| Category | Tools / Skills |
|-----------|----------------|
| **Programming** | Python, SQL, R,|
| **Libraries** | Pandas, Numpy, Scikit-learn, Matplotlib, Seaborn, ggplot2 |
| **ML / AI** | Regression, Classification, Clustering, NLP, Neural Networks, Association Rule Mining  |
| **Visualization** | Power BI|
| **Tools** | RapidMiner |

---

## 🚀 Projects

### 📊 1. Predicting factors contributing to heart attacks  
**Tools:** Python (Pandas, Matplotlib, Sckit-learn)
**Goal:** Evaluation of a logistic regression model for predicting heart attack factors using kaggle dataset. 
**Process:**
- Cleaned and transformed raw data from CSV files.  
- Built predictive model using scikit-learn. 
- Evaluated the model based on feature coefficients.
- Visualized trends and presented key insights.
  
**Result:** Accuracy 82%

🔗 ['Python code']
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

### 💬 2. Trip Advisor Hotel Reviews
**Tools:** RapidMiner (Text Mining, k-NN, Naive Bayes), Python (pandas, matplotlib)
**Goal:** Sentiment Analysis of hotel reviews in relation to rating scores using the TripAdvisor dataset from kaggle.
**Process:**
- Performed text preprocessing (tokenization, stopword removal, stemming) and vectorization (TF-IDF).
- Built a classification model (k-NN, Naive Bayes) to predict review ratings based on text features. 
- Evaluated clustering performance using Davies-Bouldin index and interpreted cluster characteristics.
- Visualized trends and presented key insights.
  
**Result:** Accuracy 91%  
🔗 [ดูโค้ดที่นี่](./Sentiment_Analysis)

---

### 📈 3. Retail Sales Dashboard (Power BI)
**Tools:** Power BI, Excel, SQL  
**Goal:** สร้าง dashboard วิเคราะห์ยอดขายและแนวโน้มลูกค้า  
**Highlight:**
- Interactive dashboard พร้อม drill-down by region  
- ใช้ DAX function เพื่อคำนวณ KPI  
🔗 [ดูภาพตัวอย่าง](./Retail_Sales_Dashboard)

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
> — Taechitpt
