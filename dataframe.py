import pandas as pd
import gensim
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression as LogReg
#from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from time import time
import os
from sklearn.metrics import classification_report
import numpy as np
#import statsmodels.api as sm

def DocumentVectors(model, model_name):
    if (model_name == "word2vec_c"):
        model_w2v = gensim.models.Doc2Vec.load_word2vec_format(model , binary=False)
        vec_vocab = [w for w in model_w2v.vocab if "_*" in w]
        vec_vocab = sorted(vec_vocab, key = lambda x: int(x[2:]))
        DocumentVectors0 = [model_w2v[w] for w in vec_vocab[:25000]]
        DocumentVectors1 = [model_w2v[w] for w in vec_vocab[25000:50000]]
    elif(model_name == "doc2vec"): #TODO
        
        try:
            model_d2v = Doc2Vec.load(model)
            
            
        except AttributeError:
            print model
            nil = np.array([0] * 25000).reshape(-1, 1)
            return (nil, nil)
        print model_d2v.docvecs.doctags
        DocumentVectors0 = [model_d2v.docvecs['SENT_'+str(i+1)] for i in range(0, 25000)]
        DocumentVectors1 = [model_d2v.docvecs['SENT_'+str(i+1)] for i in range(25000, 50000)]
    return (DocumentVectors0, DocumentVectors1)

def Classification(classifier, train, train_labels, test, test_labels):
    grid_search = GridSearchCV(classifiers_dict[classifier], param_grid = search_parameters[classifier], error_score=0.0, n_jobs = -1)
    t0 = time()
    grid_search.fit(train, train_labels)
    print("done in %0.3fs" % (time() - t0))
    #print("Best score: %0.3f" % grid_search.best_score_)
    best_parameters = grid_search.best_estimator_.get_params()
    k = ""
    for param_name in sorted(search_parameters[classifier].keys()):
        #print("%s: %r" % (param_name, best_parameters[param_name]))
        k += "%s: %r\n" % (param_name, best_parameters[param_name])
    test_prediction = grid_search.predict(test)
    test_scores = (classification_report(test_labels, test_prediction)).split('\n')
    test_score =  ' '.join(test_scores[0].lstrip().split(' ')[:-1]) +'\n' + ' '.join(test_scores[-2].split(' ')[3:-1])
    train_prediction = grid_search.predict(train)
    train_scores = (classification_report(train_labels, train_prediction)).split('\n')
    train_score =  ' '.join(train_scores[0].lstrip().split(' ')[:-1]) +'\n' + ' '.join(test_scores[-2].split(' ')[3:-1])
    return '%.3f' % grid_search.best_score_ + '\n' + 'train: ' + train_score + '\n' + 'test: ' + test_score, k[:-1]
    
    '''train_prediction = classifier.predict(train)
    train_accuracy = (100 * float(len(train_prediction[train_prediction == train_labels]))/len(train_labels))
    test_prediction = classifier.predict(test)
    test_accuracy = (100 * float(len(test_prediction[test_prediction == test_labels]))/len(test_labels))
    return 'train %.3f/test %.3f' % (train_accuracy, test_accuracy)'''
if __name__ == "__main__":
    min_count = 1# 5#TODO
    threads = 24# 20, 1#TODO
    
    d0 = ['implementation']
    parameters = ['cbow', 'size', 'alpha', 'sample', 'negative']
    min_c= ['min_count']#TODO
    classifiers = ['SklearnLogReg', 'SklearnLinearSVC']#, 'SklearnMLP'
    d3 = ['time', 'threads']
    best_params = ['best_parameters']
    df= pd.DataFrame(columns = d0+parameters+min_c+classifiers+best_params + d3)
    
    default_parameters = dict()
    classifiers_dict=dict()
    search_parameters = dict()
    time_dir = dict()
    space_dir = dict()
    
    default_parameters['size'] = 150
    default_parameters['alpha'] = 0.05
    default_parameters['window'] = 10
    default_parameters['negative'] = 25
    
    classifiers_dict['SklearnLogReg'] = LogReg()
    #classifiers_dict['SklearnMLP'] = MLPClassifier(hidden_layer_sizes = (50, 50), max_iter=1000)
    classifiers_dict['SklearnLinearSVC'] = LinearSVC()
    #classifiers_dict['StatModelsLogReg'] = sm.Logit()

    search_parameters['SklearnLogReg'] = {'solver' : ('newton-cg', 'lbfgs', 'liblinear', 'sag'), 'penalty': ('l1', 'l2'), 'dual': (False, True), 'intercept_scaling': (1, 2, 3), 'max_iter': (100, 200, 400, 800, 1000)}
    #search_parameters['SklearnMLP'] = {'solver' : ('lbfgs', 'sgd', 'adam')}
    search_parameters['SklearnLinearSVC'] = {'loss' : ('hinge', 'squared_hinge'), 'penalty': ('l1', 'l2'), 'dual': (False, True), 'intercept_scaling': (1, 2, 3),  'max_iter': (100, 200, 400, 800, 1000)}
    
    
    time_dir["word2vec_c"] = "time_w2v/"
    time_dir["doc2vec"] = "time_d2v/"
    space_dir["word2vec_c"] = "space_w2v/"
    space_dir["doc2vec"] = "space_d2v/"
    
    index  = 0
    for model_name in [ "word2vec_c"]: #TODO
        for model in os.listdir(space_dir[model_name]):
            if model.endswith('.txt'):
                string = model.split(".")[0]
                implementation = string.split()[0]
                index += 1

                df.set_value(index, 'implementation', implementation)
                df.set_value(index, 'threads', threads)#TODO
                df.set_value(index, 'min_count', min_count)#TODO

                for column in parameters:    
                    i = string.find(column)
                    
                    if (i != -1):
                        value = string[i:].split()[1]
                        df.set_value(index, column, value)
                    else:
                        if (column == 'sample'):
                            if (df.get_value(index, 'cbow') == 1):
                                df.set_value(index, column, '1e-4')
                            elif (df.get_value(index, 'cbow') == 0):
                                df.set_value(index, column, '1e-2')
                        else:
                            df.set_value(index, column, default_parameters[column])

                time_ = ""
                start = False
                for line in open(time_dir[implementation]+"time_"+model, 'r'):
                    if (not start):
                        if ("duration" in line) or ("real" in line):
                            start = True
                            time_ += line
                    else:
                        time_ += line
                    
                df.set_value(index, 'time', time_[:-1])

                DocumentVectors0, DocumentVectors1 = DocumentVectors(space_dir[model_name]+model, model_name)
                y_1 = [1] * 12500
                y_0 = [0] * 12500

                for classifier in classifiers:   
                    accuracy, best = Classification(classifier, DocumentVectors0, y_1+y_0, DocumentVectors1, y_1+y_0)
                    df.set_value(index, classifier, accuracy)
                    df.set_value(index, 'best_parameters', best)

    df.to_csv("Results.csv")
    '''parameters = {'window': (3, 5)}
    grid_search = GridSearchCV(PVClassifier(),  param_grid = parameters, cv = 2)'''
