import numpy as np
import pandas as pd
import datetime
import streamlit as st

from main import Recommender

df = pd.read_pickle('all_news_articles.pkl')

st.title('News Article Recommender')
st.image('person_reading_news.jpg', width=700)
st.write('Enter the date of publication, publication source, and article content and we will recommend an article for you.')
date = st.text_input('Date (YYYY-MM-DD)')
publication = st.text_input('Publication')
article_content = st.text_area("Article Content", height=100)
# date = st.date_input('News Date', datetime.date(2017,2,3))
# st.write(date)
if st.button('Result'):
    if date not in df.date.unique().tolist():
        st.write("Sorry! We don't have any articles with that date. Please pick another date.")
    else:
        rec = Recommender(df, date, publication, article_content)
        results = rec.recommend()

        frame = rec.article_group.drop('difference', axis=1) 
        st.write(frame)
        st.write('All Articles:', frame[frame.title != 'Your Article'].title.tolist())
        st.write('Recommended Article:')
        st.text(results)
