# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 22:40:11 2020

@author: Malcom
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys

from warnings import filterwarnings
filterwarnings("ignore")

sys.path.append(r'/opt/anaconda3/lib/python3.7/site-packages')
sys.path.append(r'/opt/anaconda3/lib/python3.7/lib-dynload')
import pandas as pd
import re
import spacy
import gensim
from gensim import corpora
import pickle
import subprocess
from gensim.models.coherencemodel import CoherenceModel
import math
import uuid

# libraries for visualization
# import pyLDAvis
# import pyLDAvis.gensim

from splunklib.searchcommands import dispatch, GeneratingCommand, Configuration, Option, validators

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
def lemmatization(texts,allowed_postags=['NOUN', 'ADJ']): 
    output = []
    for sent in texts:
        doc = nlp(sent) 
        output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
    return output

def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    """calculate the perplexity of a lda-model"""
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = [] # store the probablity of topic-word:[(u'business', 0.010020942661849608),(u'family', 0.0088027946271537413)...]
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = [] #store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0 # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0 # the num of words in the doc
        for word_id, num in dict(doc).items():
            prob_word = 0.0 # the probablity of the word 
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
    return prep

@Configuration()
class TopicModel(GeneratingCommand):
    
    test = Option(require=True, validate=validators.Boolean())
    
    start = Option(validate=validators.Integer(0))
    end = Option(validate=validators.Integer(0))
    step = Option(validate=validators.Integer(0))

    num_topics = Option(validate=validators.Integer(0))
 
    def generate(self):
        data = pd.read_csv("/opt/splunk/etc/apps/Multimodal/lookups/topic_model.csv")
        text_list= data['text'].tolist()
        
        tokens = lemmatization(text_list)
        dictionary = corpora.Dictionary(tokens)
        doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokens]
        corpora.MmCorpus.serialize('topic_modeling_test/corpus.mm', doc_term_matrix)
        
        corpus = corpora.MmCorpus('topic_modeling_test/corpus.mm')
        testset = []
        for i in range(0,corpus.num_docs,max(1,int(corpus.num_docs/20))):
            try:
                testset.append(corpus[i])
            except:
                continue
        
        # Creating the object for LDA model using gensim library
        LDA = gensim.models.ldamodel.LdaModel
        
        if self.test:
            for i in range(self.start,self.end+1,self.step):
                lda_model = LDA(corpus=corpus, 
                                id2word=dictionary, 
                                num_topics=i, 
                                random_state=100, 
                                chunksize=1000, 
                                passes=50,
                                iterations=100)
                
                coherence_model_lda = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary , coherence='c_v')
                coherence_lda = coherence_model_lda.get_coherence()
                
                prep = perplexity(lda_model, testset, dictionary, len(dictionary.keys()), i)

                yield{'Topic number': i,'Perplexity': prep, 'Coherence': coherence_lda}                
        else:
            # Build LDA model
            lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=self.num_topics, random_state=100, chunksize=1000, passes=50,iterations=100)

            lda_model.save('topic_modeling_test/lda.model')
            pickle.dump(doc_term_matrix,open('topic_modeling_test/doc_term_matrix.pkl','wb'))
            pickle.dump(dictionary,open('topic_modeling_test/dictionary.pkl','wb'))
            
            filename = uuid.uuid4().hex + '.html'

            subprocess.call(r"/opt/anaconda3/bin/python3.7 /opt/splunk/etc/apps/Multimodal/bin/topic_modeling_test/generate_vishtml.py " +filename, shell=True)

            coherence_model_lda = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary , coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            
            prep = perplexity(lda_model, testset, dictionary, len(dictionary.keys()), self.num_topics)

            yield{'filename': filename,'Perplexity': prep, 'Coherence': coherence_lda}
            

dispatch(TopicModel, sys.argv, sys.stdin, sys.stdout, __name__)
