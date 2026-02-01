# My-Portfolio
# 🧠 Taechit's Data Science Portfolio
👋 Hello! *My name's* **Taechit Khathanyaakemongkol** l *Nickname*: **Petong**

 🎂*Date of birth* : 27 July 1999 l *Age*: 26 

👔*Seeking a position in the Data Scientist/Data Analyst field where I can utilize my skills.*
   
   
---
## 🎓 Education
*M.Sc. in Data Science*, *Thammasat University (Expected to graduate : 2026)*  


---
## 🧰 Skills

| Category | Tools / Skills |
|-----------|----------------|
| **Programming Languages** | Python, SQL, R|
| **Data Science & Machine Learning** | Pandas, NumPy, Scikit-learn, Regression, Classification, Clustering, Model Evaluation, Feature Engineering|
| **NLP & Text Analytics** | Tokenization, Stopword Removal, Stemming, Lemmatization, TF-IDF, Text Classification|
| **Data Visualization** | Power BI, Matplotlib, Seaborn, ggplot2|
| **Tools & Databases** | MySQL, Streamlit(basic), Hugging Face Spaces, RapidMiner, Git/GitHub, Jupyter Notebook|
| **Statistics** | Descriptive Statistics, Hypothesis Testing, Correlation, Regression Analysis|
| **Cloud & Deployment (Basic)** | Model Deployment with Streamlit, Hugging Face Spaces, API Integration (basic)|
---

## 🚀Personal Projects

### 📊 1. Risk Factors Contributing to Heart Attack (*Academic Project*)                       
**Tools:** Python (Pandas, Matplotlib, sckit-learn).

**Goal:** Evaluation of a logistic regression model for predicting heart attack factors using kaggle dataset. 

**Process:**
- Developed and evaluated a logistic regression model to identify key risk factors associated with heart attacks using a public healthcare dataset.
- Achieved strong predictive performance through feature selection and model evaluation, supporting data-driven health insights.
  
**Result:** 
- For the positive class, the logistic regression model obtained a recall score of 88%, demonstrating its effectiveness in identifying heart attack patients.
- The risk of heart attack varies according to age, male sex, diastolic blood pressure, and heart rate.
  
🔗 [Python code_Risk Factors Contributing to Heart Attack](https://github.com/taechitpt/My_Portfolio/blob/main/Risk%20Factors%20Contributing%20to%20Heart%20Attack.ipynb)

💬 Example of Python Code
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
----------------------------------------------------------------
###  🧑‍💻2. Trip Advisor Hotel Reviews (*Academic Project*)
**Tools:** RapidMiner (Text Analytics, Classification, Clustering).

**Goal:** Sentiment analysis of hotel reviews in relation to rating scores using the TripAdvisor dataset from kaggle.

**Process:**
- Conducted sentiment analysis on hotel reviews to examine the relationship between customer sentiment and rating scores.
- Built and evaluated supervised learning models (k-NN, Naive Bayes) and performed customer segmentation using clustering techniques.
- Delivered actionable insights through data visualization and interpretation of sentiment-driven patterns.

**Result:** 
- For the clustering phase, the k-Means algorithm with k = 3 yielded the lowest Davies-Bouldin index, successfully segmenting the data into three clusters. Cluster 1 mainly represents descriptive features of the hotel, while Clusters 2 and 3 represent positive and negative customer opinions, respectively.

🔗 [Trip Advisor Hotel Reviews - REPORT](https://drive.google.com/file/d/1YeklW6qZqACHYq9q5ZXrlH1o_ljV9pUd/view?usp=drive_link)



--------------------------------------------------------------------------
###  🖥 3. Sentiment Analysis Model Deployment
**Tools:** Python (scikit-learn, streamlit), HuggingFace Spaces

**Goal:** Develop and deploy a Logistic Regression model for text sentiment classification.

**App Model:** 🔗 [Demo: Sentiment Classification App](https://huggingface.co/spaces/taechitpt/sentiment_analysis_app)

**Process:**
- Developed NLP pipeline: preprocessing, vectorization (CountVectorizer), model training.
- Evaluated model performance and optimized features.
- Deployed model as a web service using Streamlit on HuggingFace Spaces.
  
**Result:** 
- Achieved an accuracy of 70%.

💬 Example of Python Code
```python
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

df = pd.DataFrame(text, columns=["Text","Sentiment"]) 
X = df['Text']
y = df['Sentiment']
#Vectorizing Text
count_vector = CountVectorizer()
count_vector_fit= count_vector.fit_transform(X)
bag_of_word = pd.DataFrame(count_vector_fit.toarray(), columns = count_vector.get_feature_names_out()) 
print(bag_of_word)
#Train_Test_model
X_train,X_test,y_train,y_test = train_test_split(bag_of_word,y,test_size=0.3, random_state=8)
model = LogisticRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_pred,y_test) 
```




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
