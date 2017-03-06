# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

from __future__ import print_function

from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from gensim.models import Doc2Vec
import gensim.models.doc2vec
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
class PVClassifier(BaseEstimator):
    def __init__(self, dm=0, size=150, window=5, negative=25, hs=0, min_count=1, sample=1e-2, workers=1):
        self.dm = dm
        self.size = size
        self.window = window
        self.negative = negative
        self.hs = hs
        self.min_count = min_count
        self.sample = sample
        self.workers = workers
        
    def fit(self, doc_list, train_set, alldocs):
        alpha, min_alpha = (0.05, 0.001)
        train_model = Doc2Vec(dm = self.dm, size = self.size, window = self.window, negative = self.negative, hs = self.hs, min_count = self.min_count, sample = self.sample, workers = self.workers)
        train_model.build_vocab(alldocs)
        train_model.alpha, train_model.min_alpha = alpha, min_alpha
        #train_model.train(doc_list)
        
        train_targets, train_regressors = zip(*[(doc.sentiment, train_model.docvecs[doc.tags[0]]) for doc in train_set])
        train_regressors = sm.add_constant(train_regressors)
        logit = sm.Logit(train_targets, train_regressors)
        predictor = logit.fit(disp=0)
        return predictor, train_model
    def predict(self, predictor, train_model, test_docs):
        test_regressors = [train_model.docvecs[doc.tags[0]] for doc in test_docs]
        test_regressors = sm.add_constant(test_regressors)
    
        # predict & evaluate
        test_predictions = predictor.predict(test_regressors)
        return test_predictions

    def score():
        return 0


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


categories = [
    'alt.atheism',
    'talk.religion.misc',
]
# Uncomment the following to do the analysis on all the categories
#categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

from collections import namedtuple
SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
alldocs = []  # will hold all docs in original order
with open('aclImdb/alldata-id.txt', encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
        split = ['train','test','extra','extra'][line_no//25000]  # 25k train, 25k test, 25k extra
        sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # for reshuffling per pass
Clf = PVClassifier(dm=0, size=10, window=5, negative=5, hs=0, min_count=1, sample=1e-2, workers=1)
predictor, train_model = Clf.fit(doc_list, train_docs, alldocs)
Clf.predict(predictor, train_model, test_docs)
#====================================================================
data = fetch_20newsgroups(subset='train', categories=categories)
print("%d documents" % len(data.filenames))
print("%d categories" % len(data.target_names))
print()

pipeline = Pipeline([
    ('vect', PVClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'window': (3, 5),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(PVClassifier(), parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    #print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(alldocs, alldocs, alldocs) 
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
