## Sentiment Analysis 
*- 💬 Sentiment analysis was performed using the TextBlob and VADER librarys to evaluate sentiment polarity scores.*
 ```python
sentiment_1 = "I absolutely love this product; it exceeded my expectations."
sentiment_2 = "The service was fast and the staff were very friendly."
sentiment_3 = "The hotel is located near the city center and has standard facilities."
sentiment_4 = "The food was cold and didn’t taste very good."
sentiment_5 = "I was disappointed because the delivery arrived much later than promised."

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Example of setiment score code using TextBlob
sentiment_1_score = TextBlob(sentiment_1)
print(sentiment_1_score.sentiment.polarity)
sentiment_2_score = TextBlob(sentiment_2)
print(sentiment_2_score.sentiment.polarity)

# Example of setiment score code using VADER
vader_sentiment = SentimentIntensityAnalyzer()
print (vader_sentiment.polarity_scores(sentiment_1))
print (vader_sentiment.polarity_scores(sentiment_2))
```
________________________________________________________________________________
*- 📝Sentiment Analysis of Book Reviews Using Natural Language Processing Techniques.*

- ***Goal***: To classify book review texts into positive, neutral, and negative sentiment categories using natural language processing techniques.
```python
import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#Import csv file
df = pd.read_csv("book_reviews_sample.csv")
df.head()
vander_sentiment = SentimentIntensityAnalyzer()

#Cleaning data 
df["reviewText_cleaned"] = df["reviewText"].str.lower()
df["vander_sentiment_score"] = df["reviewText_cleaned"].apply(lambda x: vader_sentiment.polarity_scores(x)['compound'])

#Data Transformation
bins = [-1,-0.05,0.05,1]
label_bin = ['Negative','Neutral','Positive']
df['vander_sentiment_label'] = pd.cut(df['vander_sentiment_score'], bins=bins, labels=label_bin)
df.head()
# Data Visualization
import matplotlib.pyplot as plt
count = df['vander_sentiment_label'].value_counts()
plt.bar(count.index,count.values)
