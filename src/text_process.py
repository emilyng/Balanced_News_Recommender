import re
import nltk
import numpy as np
import pandas as pd
from textblob import TextBlob, Word
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textstat import syllable_count, lexicon_count, sentence_count, flesch_reading_ease

def get_semantic(string):
    '''
    Input: string
    Output: semantic (polarity) of string
    '''
    blob = TextBlob(string)
    semantic = blob.sentiment.polarity
    return semantic

def word_count(string):
    '''
    Input: string
    Output: number of words in string
    '''
    return len(string.split(" "))

def text_preprocess(string):
    '''
    All lowercase string, filter out anything not alphanumeric
    Input: unprocessed string
    Output: processed string
    '''
    string = string.lower()
    string = re.sub('[^a-zA-Z]+', ' ', string).strip()
    return string

def desc_len(string):
    '''
    Input: book description string
    Output: length of description
    '''
    return len(string.split(' '))

def num_unique_words(string):
    '''
    Input: book description string
    Output: number of unique words
    '''
    return len(set(string.split(' ')))

def avg_word_len(string):
    '''
    Input: book description string
    Output: average word length
    '''
    return np.mean([len(word) for word in string.split(' ')])

def get_title_data(title):
    '''
    Input: title string
    Output: title_semantic, title_word_count
    '''
    title_semantic = get_semantic(title)
    title_word_count = word_count(title)
    return title_semantic, title_word_count

def get_desc_data(string):
    '''
    Input: book description string
    Output: returns desc_len, num_unique_words, avg_word_len
    '''
    #Data before text processing
    desc_semantic = get_semantic(string)
    syl_count = syllable_count(string)
    lex_count = lexicon_count(string)
    sent_count = sentence_count(string)
    flesch = flesch_reading_ease(string)

    #Data after text processing
    string = text_preprocess(string)
    word_cnt = word_count(string)
    description_len = desc_len(string)
    number_unique_words = num_unique_words(string)
    average_word_len = avg_word_len(string)
    return desc_semantic, word_cnt, description_len, number_unique_words, \
           average_word_len, syl_count, lex_count, sent_count, flesch

stop = stopwords.words('english')
def remove_stop_words(string):
    '''
    Input: description string
    Output: description string without any stop words
    '''
    new_string = " ".join(x for x in string.split() if x not in stop)
    return new_string

def remove_most_least_freq_words(string):
    '''
    Input: description w/o stop words
    Output: stop words processed description with most and least
    frequent words removed
    '''
    most_freq = pd.Series(' '.join(string).split()).value_counts()[:10]
    least_freq = pd.Series(' '.join(string).split()).value_counts()[-10:]
    freq = list(most_freq.index).append(list(least_freq.index))
    try:
        string = " ".join(x for x in string.split() if x not in freq)
    except:
        pass
    return string

def lemmatize(string):
    '''
    Input: stop word processed string
    Output: lemmatized string
    '''
    lem_string = " ".join([Word(word).lemmatize() for word in string.split()])
    return lem_string

def prep_desc(string):
    '''
    Input: description string
    Output: return fully processed description string ready for vectorization
    '''
    prepped0 = text_preprocess(string)
    prepped1 = remove_stop_words(prepped0)
    prepped2 = remove_most_least_freq_words(prepped1)
    fully_processed  = lemmatize(prepped2)
    return fully_processed
