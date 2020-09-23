import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def article_sentiment_score(text):
    '''
    Returns average sentiment score of all sentences in article
    '''
    vader = SentimentIntensityAnalyzer()
    sentiment_score = []
    for sentence in nltk.sent_tokenize(text):
        sentiment_score.append(vader.polarity_scores(sentence)['compound'])
    return np.mean(sentiment_score)

def get_sentiment(content):
    '''
    Returns sentiment score based on entire article content
    '''
    vader = SentimentIntensityAnalyzer()
    return vader.polarity_scores(content)['compound']

def get_intro_sentiment(text):
    '''
    Returns the average sentiment of first 5 sentences in article
    '''
    vader = SentimentIntensityAnalyzer()
    sentiment_score = []
    for sentence in nltk.sent_tokenize(text)[:5]:
        sentiment_score.append(vader.polarity_scores(sentence)['compound'])
    return np.mean(sentiment_score)

def get_intro_conclusion_sentiment(text):
    '''
    Returns the average sentiment score between intro and conclusion of article
    '''
    vader = SentimentIntensityAnalyzer()
    sentiment_score = []
    first_5 = ' '.join(nltk.sent_tokenize(text)[:5])
    last_5 = ' '.join(nltk.sent_tokenize(text)[-5:])
    sentences = [first_5, last_5]
    for sentence in sentences:
        sentiment_score.append(vader.polarity_scores(sentence)['compound'])
    return np.mean(sentiment_score)
