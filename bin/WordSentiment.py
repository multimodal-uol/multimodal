#!/usr/bin/env python
# coding=utf-8
#
# Copyright © 2011-2015 Splunk, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"): you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


import sys
import os

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.data import path as nltk_data_path
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
CORPORA_DIR = os.path.join(BASE_DIR,'nltk_data')
nltk_data_path.append(CORPORA_DIR)




@Configuration()
class WordSentiment(StreamingCommand):
    def stream(self, records):
        for record in records:
            origText = record['origText']
            text = record['text']
            tokenized_sentence = nltk.word_tokenize(text)
            sid = SentimentIntensityAnalyzer()
            pos_word_list=[]
            neu_word_list=[]
            neg_word_list=[]

            for word in tokenized_sentence:
                if (sid.polarity_scores(word)['compound']) >= 0.1:
                    pos_word_list.append(word + ' '+str(sid.polarity_scores(word)['compound']))
                elif (sid.polarity_scores(word)['compound']) <= -0.1:
                     neg_word_list.append(word+ ' '+str(sid.polarity_scores(word)['compound']))
                else:
                    neu_word_list.append(word + ' '+str(sid.polarity_scores(word)['compound'])) 
            score = sid.polarity_scores(origText)
            sentiment = score['compound']
            yield{'Text': origText, 'Positive':pos_word_list, 'Neutral':neu_word_list,
                  'Negative':neg_word_list, 'Overall Score': str(score).replace(',','\n,'), 'Sentiment':sentiment}


            
dispatch(WordSentiment, sys.argv, sys.stdin, sys.stdout, __name__)


