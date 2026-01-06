## Sentiment Analysis 
- Sentiment analysis was performed using the TextBlob library to evaluate sentiment polarity scores.
 ```python
sentiment_1 = "I absolutely love this product; it exceeded my expectations."
sentiment_2 = "The service was fast and the staff were very friendly."
sentiment_3 = "The hotel is located near the city center and has standard facilities."
sentiment_4 = "The food was cold and didn’t taste very good."
sentiment_5 = "I was disappointed because the delivery arrived much later than promised."

from textblob import TextBlob
# Example of setiment score code
sentiment_1_score = TextBlob(sentiment_1)
print(sentiment_1_score.sentiment.polarity)
sentiment_2_score = TextBlob(sentiment_2)
print(sentiment_2_score.sentiment.polarity)

