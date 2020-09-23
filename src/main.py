import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from text_process import text_preprocess, remove_stop_words, lemmatize
from sentimentAnalyzer import get_intro_conclusion_sentiment

class Recommender:

    def __init__(self, data, date, publication, new_content):
        self.date = date
        self.publication = publication
        self.new_content = new_content
        self.df = data[data.date == date][['title', 'publication', 'date',
                    'content', 'content_processed']].reset_index().drop('index', axis =1)
        if len(self.df) == 0:
            print("Sorry! We don't have any articles with that date. Please pick another date.")


        #self.topic_word = None
        self.doc_topic = None
        self.doc_topic_mat = None
        self.grouped_articles = None
        self.article_group = None
        self.current_score = None

    def process_text(self):
        '''preprocessing steps for article contents'''
        txt_processed = lemmatize(remove_stop_words(text_preprocess(self.new_content)))
        return txt_processed

    def add_new_content(self):
        '''Adding new articles to cluster and base recommender on using pre-stored data'''
        if self.new_content not in self.df.content.tolist():
            content_processed = self.process_text()
            new_article = {'title': 'Your Article', 'publication': self.publication, 'date': self.date,
                           'content': self.new_content, 'content_processed': content_processed}
            self.df = self.df.append(new_article, ignore_index=True)
        else:
            row_idx = self.df[self.df.content == self.new_content].index
            self.df.loc[row_idx, 'title'] = 'Your Article'
            #self.df[self.df.content == self.new_content]['title'] = 'Your Article'

    def vectorize(self):
        '''TF-IDF vectorizer'''
        self.add_new_content()
        text = self.df['content_processed']
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(text)
        return tfidf

        '''topic_word matrix'''
#     def get_topic_word():
#         index = []
#         for n in range(n_components):
#             index.append('topic_{}'.format(n))
#         topic_word = pd.DataFrame(lsa.components_.round(3),
#                      index = index,
#                      columns = vectorizer.get_feature_names())
#         topic_word

    def get_doc_topic(self):
        '''
        Using PCA/LSA for dimensionality reduction;
        '''
        tfidf = self.vectorize()
        n_components = int(math.ceil((tfidf.shape[0]*0.75) / 5.0) * 5.0)
        lsa = TruncatedSVD(n_components, random_state=0)
        self.doc_topic_mat = lsa.fit_transform(tfidf)
        index = []
        for n in range(n_components):
            index.append('topic_{}'.format(n))

        self.doc_topic_mat = pd.DataFrame(self.doc_topic_mat, columns = index)
        self.doc_topic = self.df.join(self.doc_topic_mat)


        ''''Using LSA and maximum coefficient approach to categorize articles (deprecated)'''
        #doc_topic['topic'] = doc_topic.idxmax(axis=1).to_frame()
        #self.doc_topic['publication'] = self.df[self.df.date == self.date]['publication'].values
#         self.doc_topic['content'] = self.df['content'].values
#         self.doc_topic['content_processed'] = self.df['content_processed'].values


        #return self.doc_topic

    def get_clusters(self):
        '''
        Appends clust_topic column to doc_topic dataframe, assignment decided using agglomerative clustering
        '''
        self.get_doc_topic()
        n_components = int(math.ceil((self.df.shape[0]*0.75) / 5.0) * 5.0)
        X = self.doc_topic.select_dtypes('number').to_numpy()
        model = AgglomerativeClustering(linkage='ward', n_clusters = n_components)
        cluster_assignments = model.fit_predict(X)

        self.doc_topic['clust_topic'] = cluster_assignments

        #return self.doc_topic.reset_index()[['title', 'content', 'clust_topic']]

#     def retrieve_groups(self):
#         '''
#         Returns the indices of grouped articles
#         '''
#         self.get_clusters()
#         topic_counts = self.doc_topic['clust_topic'].value_counts()
#         topics = topic_counts[topic_counts.values > 1]

#         grouped_titles = []
#         for topic in topics.index:
#             grouped_titles.append(self.doc_topic[self.doc_topic['clust_topic'] == topic].index.tolist())

#         self.grouped_articles = grouped_titles

    def retrieve_article_group(self):
        '''
        Returns a list of grouped articles and their titles.
        '''
        self.get_clusters()
        topic_counts = self.doc_topic['clust_topic'].value_counts()
        topics = topic_counts[topic_counts.values > 1]

        grouped_titles = []
        for topic in topics.index:
            grouped_titles.append(self.doc_topic[self.doc_topic['clust_topic'] == topic].title.tolist())

        self.grouped_articles = grouped_titles

        group_num = self.doc_topic[self.doc_topic.title == 'Your Article'].clust_topic.values[0]
        article_group = self.doc_topic[self.doc_topic.clust_topic == group_num][['title', 'publication', 'content']]
        self.article_group = article_group.reset_index().drop('index', axis = 1)

    def get_sentiment(self):
        '''append sentiment score columns to article_group dataframe'''
        self.retrieve_article_group()
        ssr = self.article_group['content'].apply(get_intro_conclusion_sentiment).round(3)
        self.article_group['sentiment score (raw)'] = ssr
        self.article_group['sentiment score (scaled)'] = np.interp(ssr, (ssr.min(), ssr.max()), (-1, +1)).round(3)
        self.article_group = self.article_group.sort_values(by='sentiment score (raw)', ascending=False, ignore_index=True)


    def recommend(self):
        self.get_sentiment()

        #current article is in its own group, nothing to compare to...
        if len(self.article_group) == 1:
            return 'Unable to find articles of the same story. This article is unique!'

        #if article group only has two articles, return the other article
        elif len(self.article_group) == 2:
            return self.article_group[self.article_group.title != 'Your Article'].title.to_list()[0]

        else:
            self.current_score = self.article_group[self.article_group.title == 'Your Article']['sentiment score (scaled)'].values[0]
            diffs = [self.current_score - score for score in self.article_group['sentiment score (scaled)'].values]
            self.article_group['difference'] = diffs
            current_idx = np.where(self.article_group.difference == 0 )[0][0]

            #current article is in the middle of the polarity range, recommond both mor negative and postive articles
            if len(diffs)/current_idx == 2:
                return (self.article_group.iloc[0].title, self.article_group.iloc[-1].title)
            else:
                opp_score = list(reversed(diffs))[current_idx]
                return self.article_group.iloc[np.where(diffs == opp_score)[0][0]].title
