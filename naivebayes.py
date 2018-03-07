
"""
Naive Bayes classifier for movie review sentiment analysis
Classifying movie reviews as 'p'ositive (rating > 7 of 10) 
or 'n'egative (<4) 

Assigmnent for INFO284: Machine learning

Malte Lau Petersen
maltelau@protonmail.com
"""


import pandas as pd
import numpy as np
import scipy.sparse as sp

import re
import random
import math
import glob
import collections
import time
import string
import logging
import nltk
import pickle
import os

from typing import Tuple, List

REMOVE_PUNCTUATION_TABLE = dict((ord(char), None) for char in string.punctuation)
SENTENCE_RE = '[!"(),-.:;?[{}\]\`|~]'
NOT_RE = '(not|no|never|nev|n\'t)(.*)'



################################
# Set up logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

#################################
# Debugging test data

debug_X = pd.Series(["no no no, this is not supposed ",
                     "This movie was great!",
                     "I'd never seen a Tarzan movie ",
                     "Oh my, I think this may be the single cheesiest",
                     "I don't know what, but I like it."])
debug_Y = pd.Series(['n', 'p', 'p', 'n', 'p'])



#################################
# Text preprocessing steps

class AnalysisStep:
    """
    Base class for each analysis step in the pipeline
    __init__ is called when the pipeline is defined, so if there are any settings, 
        they should be taken here.
    fit() is called once on the training data, and doesn't change the training data
    transform() is called once on the training data, and once on the test data. 
    
    Only transform() needs to be overridden.

    """
    def __init__(self):
        pass
    def fit(self, X):
        return self
    def transform(self, X):
        raise NotImplementedError
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class RemovePunctuation(AnalysisStep):
    def __init__(self, translate_table):
        self.tt = translate_table
    def transform(self, X: pd.Series) -> pd.Series:
        logger.debug("Removing punctuation")
        return X.apply(lambda review: review.translate(self.tt))

    
class LowerCase(AnalysisStep):
    def transform(self, X: pd.Series) -> pd.Series:
        logger.debug("Transforming to lowercase")
        return X.apply(lambda s: s.lower())

    
class PrefixNot(AnalysisStep):
    def __init__(self, not_re, sentence_re):
        self.not_re = not_re
        self.sentence_re = sentence_re
        
    def transform(self, X):
        logger.debug("Prefixing NOT_")
        return X.apply(self._prefix_on_review)
        
    def _prefix_on_review(self, review):
        return ' '.join([self._prefix_on_sentence(s) for s in re.split(self.sentence_re, review)])
    
    def _prefix_on_sentence(self, sentence):
        rgx = re.search(self.not_re, sentence)
        if rgx:
            # negatives found in the sentence, prepend NOT_ to all words after that.
            return ' '.join([rgx.group(1)] + ["NOT_" + w for w in rgx.group(2).split()])
        else:
            # no negatives found
            return sentence

class LancasterStem(AnalysisStep):
    def __init__(self):
        self.st = nltk.stem.lancaster.LancasterStemmer()
    def transform(self, X):
        logger.debug("Stemming with nltk.stem.lancaster")
        return X.apply(lambda review: ' '.join([self.st.stem(word) for word in review.split()]))
        

class WordNetLemmatize(AnalysisStep):
    def __init__(self):
        self.lemma = nltk.wordnet.WordNetLemmatizer()
    def transform(self, X):
        logger.debug("Lemmatizing with nltk.wordnet")
        return X.apply(self.lemma.lemmatize)

class FilterStopwords(AnalysisStep):
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words("english")
    def transform(self, X):
        logger.debug("Filtering out stopwords from nltk.corpus.stopwords")
        return X.apply(lambda review: ' '.join([word for word in review.split()
                                                if not word in self.stopwords]))

    
class BagOfWordsEncode(AnalysisStep):
    def __init__(self):
        self.word_set = None
        self.word_set_dictionary = None

    def fit(self, X):
        logger.debug("calculating word set")
        self.word_set = list(set([word for review in X for word in review.split()]))
        self.word_set_dictionary = dict((v,k) for k,v in enumerate(self.word_set))
        return self


    def transform(self, X: pd.Series) -> sp.csr_matrix:
        if not self.word_set:
            logger.error('Need to fit the encoder before transforming')
        
        logger.debug("generating word vectors")
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
        indptr = [0]
        indices = []
        data = []
        for review in X:
            for word in review.split():
                index = self.word_set_dictionary.setdefault(word)
                if index:
                    indices.append(index)
                    data.append(1)
            indptr.append(len(indices))

        X_enc = sp.csr_matrix((data, indices, indptr),
                              shape = (len(X),
                                       len(self.word_set)))
        return X_enc

    
class BinaryBagOfWordsEncode(BagOfWordsEncode):
    def transform(self, X: pd.Series) -> sp.csr_matrix:
        for i, review in enumerate(X):
            X[i] = ' '.join(set(review.split()))

        return super().transform(X)
        


############################################
# The classifier
    
class NaiveBayes(AnalysisStep):
    """
    Fit a naive bayes classifier

    Input to .fit():
      X: sparse matrix of n * w word counts
      Y: a column of n labels (of u unique classes)
      alpha: add-alpha smoothing (add alpha to each word vector during training
                                  to avoid probabilities of zero)
    Intermediates:
      .prior: list of u prior probabilities, log(P(class))
      .likelihood: sparse matrix of u * w likelihoods, log(P(word|class))
      .posterior: matrix of n x u posteriors, log(p(class|word))

    Output:
      .predict(X) returns predicted labels for new data
    
    """      
    def fit(self,
            X: sp.csr_matrix,
            Y: pd.Series,
            alpha: float = 1):
        
        # self.X_train = X
        self.Y_train = Y
        self.classes = Y.unique()

        self.X_train_enc = X
        self.alpha = alpha
        
        # calculate the priors P(c)
        self.prior = np.log(np.array(Y.value_counts() / len(Y)))
        
        logger.debug("calculating likelihood")
        # initialize an empy array of the correct shape
        self.unstd_likelihood = np.empty([len(self.classes),
                                          X.shape[1]])

        # for each class, sum up the counts for each word
        for i, cls in enumerate(self.classes):
            idx = np.where(self.Y_train == cls)
            self.unstd_likelihood[i] = self.X_train_enc[idx].\
                                       sum(axis = 0) \
                                       + alpha

        self.likelihood = np.log(
            self.unstd_likelihood \
            / np.apply_along_axis(sum, 0, self.unstd_likelihood))
        
        return self


    def predict(self, X: sp.csr_matrix) -> pd.Series:
        # return the predicted labels (ie y_pred)

        logger.debug("calculating posterior")
        self.posterior = np.hstack([(X * sp.diags(cls)).sum(axis = 1) \
                                    for cls in self.likelihood]) \
                                        + self.prior
        
        # select the class with the highest posterior
        self.predictions = pd.Series(self.classes[[np.argmax(review) \
                                                            for review in self.posterior]])
        return self.predictions


###############################################
# Helper functions


def contingency(true, predicted, pos):
    tp = sum([(t == pos) & (pos == p) for t,p in zip(true, predicted)])
    tn = sum([(t != pos) & (pos != p) for t,p in zip(true, predicted)])
    fp = sum([(t != pos) & (pos == p) for t,p in zip(true, predicted)])
    fn = sum([(t == pos) & (pos != p) for t,p in zip(true, predicted)])

    return tp,tn,fp,fn
    

def F1(true, predicted, pos, beta = 1):
    tp,tn,fp,fn = contingency(true, predicted, pos)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    if tp == 0: return 0
    else:
        return ((beta ^ 2) + 1) * precision * recall / ((beta ^ 2) * precision + recall)
    
def accuracy(true, predicted, pos):
    tp,tn,fp,fn = contingency(true, predicted, pos)
    acc = (tp + tn) / (tp + fp + tn + fn)
    return acc

    
def brier_score(Y, classes, posterior):
    # calculate the brier score (a commonly used proper scoring rule)
    # 1 = worst
    # 0 = best
    N = len(Y)
    true_classes = np.array(Y.apply(lambda cls: int(np.where(classes != cls)[0])))
    posterior = np.exp(posterior - mean(posterior, axis = 1).reshape(N, 1))
    posterior = (posterior / posterior.sum(axis = 1).reshape(N, 1))

    term = posterior[:,0] - true_classes
    brier = sum(term ** 2) / N

    return brier
    
    
def cv_folds(n, k):
    # generator that yields indices used to subset the data for crossvalidation
    if n % k == 0:
        pad = 0
    else:
        pad = 1
    proposal = [i for i in range(n)]
    random.shuffle(proposal)
    size = math.ceil(n / k)
    for i in range(n//size+pad):
        yield proposal[size*i:size*(i+1)]
    
    
def read_data(path, max_files = None):
    # read through the data in DATA/
    X = pd.Series()
    Y = pd.Series()
    l = 0
    for c,cls,k in zip(['p', 'n'], ['pos', 'neg'], [0,1]):
        files = glob.glob(path + cls + "/*.txt")
        if max_files: files = files[:max_files]

        for i, name in enumerate(files):
            i = i + l * k
            l = len(files)
        
            idx, rest = name.split("/")[-1].split("_")
            rating, _ = rest.split(".")

            with open(name) as f:
                txt = f.read()

                X.loc[i] = txt
                Y.loc[i] = c
    return X, Y


def save_model(model, pipeline, folder = "model"):
    """ Save a model to disk. """
    
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass

    with open(os.path.join(folder, "pipeline.pickled"), "wb") as f:
        pickle.dump(pipeline, f)

    with open(os.path.join(folder, "model.pickled"), "wb") as f:
        pickle.dump(model, f)
    


def load_model(folder = "model"):
    """ Load a model from disk. """
    
    with open(os.path.join(folder, "pipeline.pickled"), "rb") as f:
        pipeline = pickle.load(f)

    with open(os.path.join(folder, "model.pickled"), "rb") as f:
        model = pickle.load(f)

    return model, pipeline
    

###################################3
# Functions to run the analysis


def preprocess_pipeline(pipeline: List[AnalysisStep],
                        train: pd.Series, test: pd.Series = None) -> Tuple[sp.csr_matrix]:
    
    for i in range(len(pipeline)):
        logger.info(f'Preprocessing pipeline step {i+1} of {len(pipeline)}')
        pipeline[i] = pipeline[i].fit(train)
        train = pipeline[i].transform(train)
        if not test is None:
            test = pipeline[i].transform(test)

    return train, test


def run_once(X_train, X_test, Y_train, Y_test, pipeline, model):
    X_train, X_test = preprocess_pipeline(pipeline, X_train, X_test)
    model = model.fit(X_train, Y_train, alpha = 1)
    Y_pred = model.predict(X_test)
    logger.info(f"""
Prediction counts: {dict(collections.Counter(Y_pred))}
F1: {F1(Y_test, Y_pred, 'p'):.3f}
Accuracy: {accuracy(Y_test, Y_pred, 'p'):.3f}
Brier score: {brier_score(Y_test, model.classes, model.posterior):.3f}""")
    return model

    

def run_cv(X, Y, models):
    results = pd.DataFrame(columns = ['Model', 'Pipeline version','k',
                                      'n', 'Accuracy', 'F1', 'Brier', 'Time'])

    for j, (pipeline, model) in enumerate(models):

        model_name = type(model).__name__
        logger.info(f"Model: {model_name}, Pipeline Version: {j+1}")

        X_pre, _ = preprocess_pipeline(pipeline, X)
    
        for k in [10]: # k fold cv
            logger.info(f"Running {k} fold cross validation")

            Y_TEST = Y
            Y_PRED = pd.Series([None] * len(Y))
            POSTERIOR = np.array(np.empty((len(Y), len(Y.unique()))))
        
            starttime = time.time()
            for it, fold in enumerate(cv_folds(len(Y), k)):
                # process one fold
                logger.info(f"Fold {it+1} of {k}")
                rest = [i for i in range(len(X_train)) if i not in fold]
                x_train = X_pre[rest,:]
                y_train = Y.iloc[rest]
                
                x_test = X_pre[fold,:]
                y_test = Y.iloc[fold]
                
                model = model.fit(x_train, y_train)
                # store the predictions for this fold in the right place in Y_PRED
                Y_PRED[fold] = model.predict(x_test)
                POSTERIOR[fold] = model.posterior
                
                del x_train, y_train, x_test, y_test

            # now calculate f1 and accuracy for the predictions from all folds
            f1=F1(Y_TEST, Y_PRED, "p")
            acc=accuracy(Y_TEST, Y_PRED, "p")
            t=time.time()-starttime
            brier = brier_score(Y_TEST, Y.unique(), POSTERIOR)
            results.loc[len(results)+1] = [model_name, j+1, k, len(Y_train), acc, f1, brier, t]
            
        del model, X_pre, _
            
    logger.info('\n' + str(results))
    return results






# if __name__ == '__main__':
if True:

    
    logger.info("loading data")
    # load and parse the data as it were originally
    # this is slow ...
    # X_train, Y_train = read_data("DATA/aclImdb/train/", 1000)
    # X_test, Y_test = read_data("DATA/aclImdb/test/", 1000)
    
    # .. or load it once and save all in a csv
    # X_train.to_csv("X_train.csv")
    # X_test.to_csv("X_test.csv")
    # Y_train.to_csv("Y_train.csv")
    # Y_test.to_csv("Y_test.csv")
    
    X_train = pd.read_csv("X_train.csv").text
    X_test = pd.read_csv("X_test.csv").text
    Y_train = pd.read_csv("Y_train.csv", header = None).loc[:,1]
    Y_test = pd.read_csv("Y_test.csv", header = None).loc[:,1]
    
    logger.info("data loaded")



    # a model is (pipeline, model) where pipeline is a l
    # haven't found a more elegant way to specify these ..
    # models =  [\
    #         ([RemovePunctuation(REMOVE_PUNCTUATION_TABLE), #1
    #           LowerCase(),
    #           BinaryBagOfWordsEncode()],
    #          NaiveBayes()),
            
    #         ([RemovePunctuation(REMOVE_PUNCTUATION_TABLE), #2
    #           LowerCase(),
    #           PrefixNot(NOT_RE, SENTENCE_RE),
    #           BinaryBagOfWordsEncode()],
    #          NaiveBayes()),
            
    #         ([RemovePunctuation(REMOVE_PUNCTUATION_TABLE), #3
    #           LowerCase(),
    #           BagOfWordsEncode()],
    #          NaiveBayes()),
            
    #         ([RemovePunctuation(REMOVE_PUNCTUATION_TABLE), #4
    #           LowerCase(),
    #           PrefixNot(NOT_RE, SENTENCE_RE),
    #           BagOfWordsEncode()],
    #          NaiveBayes())]

    models = [
            ([RemovePunctuation(REMOVE_PUNCTUATION_TABLE), #3
              LowerCase(),
              BagOfWordsEncode()],
             NaiveBayes()),
            ([RemovePunctuation(REMOVE_PUNCTUATION_TABLE), #3
              LowerCase(),
              FilterStopwords(),
              BagOfWordsEncode()],
             NaiveBayes())]
    
    results = run_cv(X_train, Y_train, models)
    # now select the best pipeline from these results and test it against the test set:

    best_model = results.sort_values("Brier").loc[:,"Pipeline version"].head(1)
    
    best_pipeline = models[int(best_model)-1]
    logger.info(f"Best pipeline + model:")
    for p in best_pipeline[0]:
        logger.info('+ ' + type(p).__name__)
    logger.info('* ' + type(best_pipeline[1]).__name__)



    
    logger.info("Fitting best model on full training set and testing against test set")
    model = run_once(X_train, X_test, Y_train, Y_test, best_pipeline[0], best_pipeline[1])


    
    logger.info("Fitting best model on training + test set")
    X = pd.concat((X_train, Y_train))
    X, _ = preprocess_pipeline(best_pipeline[0], X, None)
    Y = pd.concat((Y_train, Y_test))
    model = best_pipeline[1].fit(X, Y, alpha = 1)
    save_model(model, best_pipeline[0])



