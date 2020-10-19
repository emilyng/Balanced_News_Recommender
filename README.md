# Balanced News Recommender

## Introduction
We are living in an age where we have a wealth of information at our fingertips.

The two biggest problems in this age of information are misinformation and filter bubbles.
Tech giants like Google and Facebook develop algorithms that help people find content that they would be interested in, and most often than not people are more interested in things that they already align with. However, there is a difference between what people want to hear and what people need to hear.

This creates a filter bubble in which people are unable to understand and communicate with others of different beliefs, thereby hindering any positive change through compromise. This effect is most prominently seen in politics, where we see more of a divide between people and ideologies every day.

Some companies that tackle filter bubbles are [AllSides](https://www.allsides.com/unbiased-balanced-news) and [FlipSide](https://www.theflipside.io/).

AllSides calculates a media bias rating of  articles ranging from Left leaning to Right leaning.
This media bias rating is calculated with a combination of editorial and independent reviews and having users participate in blind bias surveys.

 FlipSide uses a similar approach by having an editorial team curate unbiased articles to share.

Both methods are time consuming and labor intensive. Is there a way to make it easier for people to get both sides of a story?

The goal of this project is to create a system that recommends articles of the opposite polarity to the one someone is already looking at.

## Data
Data used for this project comes from [Kaggle](https://www.kaggle.com/snapcrack/all-the-news).

## Documentation
This project contains work done in Jupyter Notebooks and .py files.

### Notebooks
Data taken from ![Kaggle](https://www.kaggle.com/snapcrack/all-the-news) is broken into 3 separate files. `News.ipynb` contains code that compiles all the data. The complete data includes articles written from years 2011 to 2018, but my project only takes in years 2015-2018 because of the lack of articles in years 2014 and before. `News.ipynb` also cleans and performs all preprocessing steps on the text data. This notebook also contains preliminary tests on different topic modeling techniques including LSA, NMF, and LDA.

`News Topic Modeling.ipynb` contains code that performs topic modeling using LSA and clusters articles of the same topic/event using Agglomerative Clustering technique.

### src
Main code for the recommender is located in `main.py`.

##### Dependencies:
- math
- numpy
- pandas
- sklearn
- nltk

`main.py` takes in some functions written in `text_process.py` and `sentimentAnalyzer.py`.

A streamlit webapp is made from `news_recommender_app.py`.

#### `sentimentAnalyzer.py`
Includes different methods of analyzing sentiment of an article. 
