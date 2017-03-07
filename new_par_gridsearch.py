from sklearn.linear_model import LogisticRegression as LogReg
from gensim.models import Doc2Vec
from sklearn.model_selection import GridSearchCV
import gensim.models.doc2vec
from sklearn.utils.fixes import signature
import warnings
from sklearn.externals import six

class PVClassifier():
    def __init__(self, dm=0, size=1, window=1, negative=1, hs=0, min_count=1, sample=1e-2, workers=1):
            self.dm = dm
            self.size = size
            self.window = window
            self.negative = negative
            self.hs = hs
            self.min_count = min_count
            self.sample = sample
            self.workers = workers

    def fit(self, train_docs):
            new_docs = []
            for line_no, line in enumerate(train_docs):
                words = line.words
                tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
                split = line.split  # 25k train, 25k test, 25k extra
                sentiment = line.sentiment # [12.5K pos, 12.5K neg]*2 then unknown
                new_docs.append(SentimentDocument(words, tags, split, sentiment))
            alpha, min_alpha = (0.05, 0.001)
            train_model = Doc2Vec(dm = self.dm, size = self.size, window = self.window, negative = self.negative, hs = self.hs, min_count = self.min_count, sample = self.sample, workers = self.workers)
            train_model.build_vocab(new_docs)
            train_model.alpha, train_model.min_alpha = alpha, min_alpha
            train_model.train(new_docs)
            print (len(new_docs))
            #exit()
            print (len(train_model.docvecs))
            print ("here")
            y = [doc.sentiment for doc in new_docs]
            regr = LogReg()
            predictor = regr.fit(train_model.docvecs, y)
            
            #train_targets, train_regressors = zip(*[(doc.sentiment, train_model.docvecs[doc.tags[0]]) for doc in train_set])
            #train_regressors = sm.add_constant(train_regressors)
            
            self.predictor = predictor
            self.train_model = train_model
            return self

    def predict(self, test_docs):
        #test_regressors = [self.train_model.docvecs[doc.tags[0]] for doc in test_docs]
        #test_regressors = sm.add_constant(test_regressors)
        alpha, min_alpha = (0.05, 0.001)
        #test_model = Doc2Vec(dm = self.dm, size = self.size, window = self.window, negative = self.negative, hs = self.hs, min_count = self.min_count, sample = self.sample, workers = self.workers)
        #print (len(test_model.docvecs))
        #test_model.build_vocab(test_docs)
        #test_model.alpha, test_model.min_alpha = alpha, min_alpha
        model = self.train_model
        print (len(model.docvecs))
        print ("there")
        model.train(test_docs)
        print (len(model.docvecs))
        print ("now")
        
        # predict
        test_predictions = self.predictor.predict(model.docvecs)
        return test_predictions

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def score(self, X):
        y = [doc.sentiment for doc in X]
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))



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
#Clf = PVClassifier()
#Clf.fit(train_docs)
#print (Clf.score(test_docs))
parameters = {'window': (3, 5)}
grid_search = GridSearchCV(PVClassifier(),  parameters)
grid_search.fit(train_docs)
