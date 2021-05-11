# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:46:21 2020

@author: sgmjin2
"""
import sys
import re
import os
import logging, logging.handlers

import warnings
warnings.filterwarnings('ignore') 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(r'/opt/anaconda3/lib/python3.7/site-packages')
sys.path.append(r'/opt/anaconda3/lib/python3.7/lib-dynload')


import pandas as pd
import joblib
import numpy as np
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.data import path as nltk_data_path
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import string
from textblob import TextBlob

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from splunk import setupSplunkLogger
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
CORPORA_DIR = os.path.join(BASE_DIR,'nltk_data')
nltk_data_path.append(CORPORA_DIR)

labels=['ARTS', 'ARTS & CULTURE', 'BLACK VOICES', 'BUSINESS', 'COLLEGE',
       'COMEDY', 'CRIME', 'CULTURE & ARTS', 'DIVORCE', 'EDUCATION',
       'ENTERTAINMENT', 'ENVIRONMENT', 'FIFTY', 'FOOD & DRINK',
       'GOOD NEWS', 'GREEN', 'HEALTHY LIVING', 'HOME & LIVING', 'IMPACT',
       'LATINO VOICES', 'MEDIA', 'MONEY', 'PARENTING', 'PARENTS',
       'POLITICS', 'QUEER VOICES', 'RELIGION', 'SCIENCE', 'SPORTS',
       'STYLE', 'STYLE & BEAUTY', 'TASTE', 'TECH', 'TRAVEL', 'WEDDINGS',
       'WEIRD NEWS', 'WELLNESS', 'WOMEN', 'WORLD NEWS', 'WORLDPOST']


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()
my_sw = ['make', 'amp',  'news','new' ,'time', 'u','s', 'photos',  'get', 'say']
def black_txt(token):
    return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2 and token not in my_sw

def clean_txt(text):
  clean_text = []
  clean_text2 = []
  text = re.sub("'", "",text)
  text=re.sub("(\\d|\\W)+"," ",text)    
  clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
  clean_text2 = [word for word in clean_text if black_txt(word)]
  return " ".join(clean_text2)


def load_obj(name):
    with open(name, 'rb') as f:
        return joblib.load(f)

def polarity_txt(text):
  return TextBlob(text).sentiment[0]

def subj_txt(text):
  return  TextBlob(text).sentiment[1]

def len_text(text):
  if len(text.split())>0:
         return len(set(clean_txt(text).split()))/ len(text.split())
  else:
         return 0

def get_multi_input(clean_text):
    polarity=polarity_txt(clean_text)
    subj=subj_txt(clean_text)
    length=len_text(clean_text)
    data ={'polarity':[polarity],
           'subj':[subj],
           'length':[length]}
    
    df = pd.DataFrame (data, columns=['polarity','subj','length'])
    return df


@Configuration(local=True)
class Classification(StreamingCommand):
    textfield = Option(
        require=True,
        doc='''
        **Syntax:** **textfield=***<fieldname>*
        **Description:** Name of the field that will contain the text to search against''',
        )

    model = Option(
        require=True,
        doc='''
        **Syntax:** **textfield=***<fieldname>*
        **Description:** Name of the model for prediction use''',
        )
        
    text_weight = Option(
        default=50,
        doc='''**Syntax:** **text_weight=***<int>*
        **Description:** The weight of text when analysing combined image and text.''',
        validate=validators.Integer()
        ) 	

    
    def stream(self, records):
        if '/' not in self.textfield:
            if(self.model=='avg_vec.mdl'):     
                #load the model
                Model=load_obj(os.path.join(os.getcwd(),'models',self.model))
                #load the word embedding
                word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
                word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True,limit=500000)
                
                for record in records:
                    clean_text=clean_txt(record[self.textfield])
                    tokenizer = RegexpTokenizer(r'\w+')
                    words=tokenizer.tokenize(clean_text)
                    average_vec=get_average_word2vec(words,word2vec)
                    record['Predict']=Model.predict([average_vec])[0]
                    yield record
            
            else:
                #load the model and word2idx
                Model=load_model(os.path.join(os.getcwd(),'models',self.model))
                word2idx = load_obj(os.path.join(os.getcwd(),'models','word2idx.pkl'))
                
                for record in records:
                    clean_text=clean_txt(record[self.textfield])
                    tokenizer = RegexpTokenizer(r'\w+')
                    words=tokenizer.tokenize(clean_text)
                    indexes = [word2idx[word] for word in words if word in word2idx]
                    input_data = pad_sequences([indexes], maxlen=60, value=29942)
                    if(self.model=='LSTM_Simple.h5'):
                        res=Model.predict(input_data)
                    else:
                        polarity=polarity_txt(clean_text)
                        subj=subj_txt(clean_text)
                        length=len_text(clean_text)
                        data ={'polarity':[polarity],
                              'subj':[subj],
                              'length':[length]}
                        
                        df = pd.DataFrame (data, columns=['polarity','subj','length'])
                        res=Model.predict([input_data[0].reshape(1,60),df])      
                        
                    record['Predict']=labels[np.argmax(res)]    
                    yield record
        else:
            text=self.textfield.split('/')[0]
            image=self.textfield.split('/')[1]
            
            if(self.model=='avg_vec.mdl'):     
                #load the model
                Model=load_obj(os.path.join(os.getcwd(),'models',self.model))
                #load the word embedding
                word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
                word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True,limit=500000)
                
                for record in records:
                    clean_text=clean_txt(record[text])
                    clean_image=clean_txt(record[image])
                    
                    tokenizer = RegexpTokenizer(r'\w+')
                    
                    words_text=tokenizer.tokenize(clean_text)
                    words_image=tokenizer.tokenize(clean_image)
                    
                    average_vec_text=get_average_word2vec(words_text,word2vec)
                    average_vec_image=get_average_word2vec(words_image,word2vec)
                    
                    confidence_text=Model.predict_proba([average_vec_text])
                    confidence_image=Model.predict_proba([average_vec_image])
                    
                    confidence_combined = self.text_weight / 100.0 * confidence_text + (100.0 - self.text_weight) / 100.0 * confidence_image
                    record['Text_Predict']=labels[np.argmax(confidence_text)]
                    record['Image_Predict']=labels[np.argmax(confidence_image)]
                    record['Predict']=labels[np.argmax(confidence_combined)]
                    yield record
            
            else:
                #load the model and word2idx
                Model=load_model(os.path.join(os.getcwd(),'models',self.model))
                word2idx = load_obj(os.path.join(os.getcwd(),'models','word2idx.pkl'))
                
                for record in records:
                    clean_text=clean_txt(record[text])
                    clean_image=clean_txt(record[image])
                    
                    tokenizer = RegexpTokenizer(r'\w+')
                    
                    words_text=tokenizer.tokenize(clean_text)
                    words_image=tokenizer.tokenize(clean_image)
                    
                    indexes_text = [word2idx[word] for word in words_text if word in word2idx]
                    indexes_image = [word2idx[word] for word in words_image if word in word2idx]
                    
                    input_data_text = pad_sequences([indexes_text], maxlen=60, value=29942)
                    input_data_image = pad_sequences([indexes_image], maxlen=60, value=29942)
                    
                    if(self.model=='LSTM_Simple.h5'):
                        confidence_text=Model.predict(input_data_text)
                        confidence_image=Model.predict(input_data_image)
                    else:
                        df_text = get_multi_input(clean_text)
                        df_image = get_multi_input(clean_image)
                        
                        confidence_text=Model.predict([input_data_text[0].reshape(1,60),df_text])
                        confidence_image=Model.predict([input_data_image[0].reshape(1,60),df_image])
                        
                    confidence_combined = self.text_weight / 100.0 * confidence_text + (100.0 - self.text_weight) / 100.0 * confidence_image
                    record['Text_Predict']=labels[np.argmax(confidence_text)]
                    record['Image_Predict']=labels[np.argmax(confidence_image)]
                    record['Predict']=labels[np.argmax(confidence_combined)]   
                    yield record

dispatch(Classification, sys.argv, sys.stdin, sys.stdout, __name__)
