
import pandas as pd
import numpy as np
# import nltk
import re
import random
import math
import glob
import collections
import scipy.sparse as sp
# import time
import string
import logging

# lemma = nltk.wordnet.WordNetLemmatizer()
translate_table = dict((ord(char), None) for char in string.punctuation)  

from typing import List

################################
# Set up logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

#################################
# Debugging test data

debug_X = pd.Series(["no no no, this is not supposed ",
                     "This movie was great!",
                     "I'd never seen a Tarzan movie ",
                     "Oh my, I think this may be the single cheesiest"])
debug_Y = pd.Series(['n', 'p', 'p', 'n'])



#################################
# Classes

class AnalysisStep:
    def __init__(self):
        pass
    def fit(self, *args, **kwargs):
        return self
    def transform(self, *args, **kwargs):
        raise NotImplementedError
    def fit_transform(self, X, *args, **kwards):
        return self.fit(X).transform(X)

class RemovePunctuation(AnalysisStep):
    def __init__(self, translate_table):
        self.tt = translate_table
    def transform(self, X: pd.Series) -> pd.Series:
        return X.apply(lambda review: review.translate(self.tt))

class LowerCase(AnalysisStep):
    def transform(self, X: pd.Series) -> pd.Series:
        return X.apply(lambda s: s.lower())

class BagOfWordsEncode(AnalysisStep):
    def __init__(self):
        self.word_set = None
        self.word_set_dictionary = None

    def fit(self, X):
        logger.info("calculating word set")
        self.word_set = list(set([word for review in X for word in review.split()]))
        self.word_set_dictionary = dict((v,k) for k,v in enumerate(self.word_set))
        return self


    def transform(self, X: pd.Series) -> sp.csr_matrix:
        if not self.word_set:
            logger.error('Need to fit the encoder before transforming')
        
        logger.info("generating word vectors")
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
        

class NaiveBayes(AnalysisStep):
    """
    Fit a naive bayes classifier
    Requires a subclass to define the encode method to transform 
      the documents (X) to word vectors (X_enc)
    Input to self.fit():
      X: sparse matrix of n * w word counts
      Y: a "list" of n labels (of u unique classes)
      alpha: add-alpha smoothing (add alpha to each word vector during training
                                  to avoid probabilities of zero)
    Intermediates:
      self.prior: list of u prior log(probabilities), p(class)
      self.likelihood: sparse matrix of u * w log(likelihood), p(word|class)
      self.posterior: matrix of n x u log(posterior), p(class|word)
    Output:
      self.predict(Y) returns predicted labels for new data Y
    
    """      
    def fit(self,
            X: sp.csr_matrix,
            Y: pd.Series,
            alpha: float = 1):
        
        # self.X_train = X
        self.Y_train = Y
        logger.info("fitting model: " + type(self).__name__)

        self.X_train_enc = X
        self.alpha = alpha
        
        # calculate the priors P(c)
        self.prior = np.log(np.array(Y.value_counts() / len(Y)))
        
        logger.info("calculating likelihood")
        # initialize an empy array of the correct shape
        self.unstd_likelihood = np.empty([len(Y_train.unique()),
                                          X.shape[1]])

        # for each class, sum up the counts for each word
        for i, cls in enumerate(Y_train.unique()):
            self.unstd_likelihood[i] = self.X_train_enc[np.where(Y_train == cls)].\
                                       sum(axis = 0) \
                                       + alpha

        self.likelihood = np.log(
            self.unstd_likelihood \
            / np.apply_along_axis(sum, 0, self.unstd_likelihood))
        
        return self


    def predict(self, X: sp.csr_matrix) -> pd.Series:
        # return the predicted labels (ie y_pred)

        logger.info("calculating posterior")
        self.posterior = np.hstack([(X * sp.diags(cls)).sum(axis = 1) \
                                    for cls in self.likelihood]) \
                                        + self.prior
        
        # select the class with the highest posterior
        self.predictions = pd.Series(self.Y_train.unique()[[np.argmax(review) \
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
    # print(tp,tn,fp,fn, recall,precision)

    if tp == 0: return 0
    else:
        return ((beta ^ 2) + 1) * precision * recall / ((beta ^ 2) * precision + recall)
    
def accuracy(true, predicted, pos):
    tp,tn,fp,fn = contingency(true, predicted, pos)
    acc = (tp + tn) / (tp + fp + tn + fn)
    return acc

def cv_folds(n, k):
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
    # X = pd.DataFrame(columns = ['id', 'rating', 'text'])
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



def run_once(X_train, X_test, Y_train, Y_test, pipeline, model):
    for i in range(len(pipeline)):
        logger.debug(f'Preprocessing pipeline step {i+1} of {len(pipeline)}')
        pipeline[i] = pipeline[i].fit(X_train)
        X_train = pipeline[i].transform(X_train)
        X_test = pipeline[i].transform(X_test)

    model = model.fit(X_train, Y_train, alpha = 1)
    Y_pred = model.predict(X_test)
    print(f"Prediction counts: {dict(collections.Counter(Y_pred))}\nF1: {F1(Y_test, Y_pred, 'p'):.3f}\nAccuracy: {accuracy(Y_test, Y_pred, 'p'):.3f}")
    return model

    

def run_cv(X_train, X_test, Y_train, Y_test):
    results = pd.DataFrame(columns = ['Model','fold', 'k', 'n', 'Accuracy', 'F1', 'Time'])
    
    for k in [5, 10]: # k fold cv
        # f1 = collections.defaultdict(list)
        # acc = collections.defaultdict(list)
        # t = collections.defaultdict(list)
        
        for it, fold in enumerate(cv_folds(len(X_train), k)): # process one fold
            logger.info(f"FOLD: {it+1} of {k}")
            rest = [i for i in range(len(X_train)) if i not in fold]
                
            x_train = X_train.iloc[rest]
            y_train = Y_train.iloc[rest]
                
            x_test = X_train.iloc[fold]
            y_test = Y_train.iloc[fold]
            
            for pipeline, model in [([
                    RemovePunctuation(translate_table),
                    LowerCase(),
                    BinaryBagOfWordsEncode()],NaiveBayes()),([
                        RemovePunctuation(translate_table),
                        LowerCase(),
                        BagOfWordsEncode()], NaiveBayes())]:
                model_name = type(model).__name__
                print("Model:", model_name)
                starttime = time.time()
               
                # print("training word vectorizer")
                # feature_encoder = countvec()
                # feature_encoder.fit(x_train)
                
                y_pred = run_once(x_test, x_test, y_train, y_test, pipeline, model)
                # y_pred = model().fit(x_train, y_train).predict(x_test)
                # print(y_pred)
                f1=F1(y_test, y_pred, "p")
                acc=accuracy(y_test, y_pred, "p")
                t=time.time()-starttime
                

                # print("folds", fold)
                # print(fold, [x for x in y_pred], [x for x in y_test], F1(y_test, y_pred, "p"))
                
                
                results.loc[len(results)+1] = [model_name, it+1, k, len(Y_train), acc, f1, t]
                # print("Results for the model", model)
                # print("F1 Score:", "{:.2f}".format(mean(f1)))
                # print("Accuracy:", "{:.2f}".format(mean(acc)))
                # print(f1,acc)
    # print(results.groupby("Model")['Accuracy', 'F1'].mean())
    print(results)
    return results


logger.info("loading data")
# X_train, Y_train = read_data("DATA/aclImdb/train/", 1000)
# X_test, Y_test = read_data("DATA/aclImdb/test/", 1000)

# X_train.to_csv("X_train.csv")
# X_test.to_csv("X_test.csv")
# Y_train.to_csv("Y_train.csv")
# Y_test.to_csv("Y_test.csv")

X_train = pd.read_csv("X_train.csv").text
X_test = pd.read_csv("X_test.csv").text
Y_train = pd.read_csv("Y_train.csv", header = None).loc[:,1]
Y_test = pd.read_csv("Y_test.csv", header = None).loc[:,1]

logger.info("data loaded")
# run_cv()


# pipeline = [
#     RemovePunctuation(translate_table),
#     LowerCase(),
#     BinaryBagOfWordsEncode()
# ]
# model = NaiveBayes()

# model = run_once(X_train, X_test, Y_train, Y_test, pipeline, model)
results = run_cv(X_train, X_test, Y_train, Y_test)


# wv = BagofWords_encoder()
# X_train_enc = wv.fit_transform(X_train)
# print(X_train_enc)

# from  sklearn.feature_extraction.text import CountVectorizer 
# from sklearn.naive_bayes import MultinomialNB
# # 
# def run_sk():
#     print("constructing word vectors")
#     bow = CountVectorizer()
#     X_train_enc = bow.fit_transform(X_train.text)
#     X_test_enc = bow.transform(X_test.text)
#     print("fitting model")
#     skNB = MultinomialNB()
#     skNB.fit(X_train_enc, Y_train)
#     print("predicting labels")
#     y_pred = skNB.predict(X_test_enc)
#     print("calculating F1")
#     print(F1(Y_test, y_pred, "p"))


# run_sk()
#     print("sklearn F1:", F1(Y_test, y_pred, "p"))

# def run_custom():
#     mod = BagOfWordsNB().fit(X_train.text, Y_train)
#     y_pred = mod.predict(X_test.text)
#     print("custom F1:", F1(Y_test, y_pred, "p"))

# class BagOfWordsNB_cv(BagOfWordsNB):
#     def lemmatize(self, X):
#         return X
        
# run_custom()

# import timeit
# print(timeit.timeit('run_sk()' ,setup = 'from __main__ import run_sk', number = 2))
# print(timeit.timeit('run_custom()' ,setup = 'from __main__ import run_custom', number = 2))








