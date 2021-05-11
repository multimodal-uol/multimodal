# libraries for visualization
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sys
    import pyLDAvis
    import pyLDAvis.gensim
    import gensim
    import pickle

sys.path.append(r'/opt/anaconda3/lib/python3.7/site-packages')
sys.path.append(r'/opt/anaconda3/lib/python3.7/lib-dynload')

BASE_DIR = '/opt/splunk/etc/apps/Multimodal/bin'
filename = sys.argv[1]
LDA = gensim.models.ldamodel.LdaModel
newlda = LDA.load(BASE_DIR+'/topic_modeling_test/lda.model')
newdoc_term_matrix=pickle.load(open(BASE_DIR+'/topic_modeling_test/doc_term_matrix.pkl', 'rb'))
newdictionary=pickle.load(open(BASE_DIR+'/topic_modeling_test/dictionary.pkl', 'rb'))
vis = pyLDAvis.gensim.prepare(newlda, newdoc_term_matrix, newdictionary, mds='mmds')
pyLDAvis.save_html(vis, '/opt/splunk/etc/apps/Multimodal/appserver/static/reports/'+filename)
#pyLDAvis.save_html(vis, BASE_DIR+'/topic_modeling_test/'+filename)