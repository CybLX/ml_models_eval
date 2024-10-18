
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from torch.utils.data import Dataset
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import torch

#######################################################
#               Remocao de outliers
#######################################################

class RemoveOutliers(object):
    
    def __init__(self, ClassTogether = True, target = 'comentario'):
        self.ClassTogether = ClassTogether
        self.target = target
        self.data = None
    
    @staticmethod
    def outliers(df, cols):
        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df


    def statistics(self,data):        
        if self.ClassTogether:
            data = data.copy()
            data['char_count'] = data[self.target].apply(len)
            data['word_count'] = data[self.target].apply(lambda x: len(x.split()))
            data = RemoveOutliers.outliers(df = data, cols = ['char_count', 'word_count'])
            data['word_density'] = data['char_count'] / (data['word_count'] + 1)
            #data['punctuation_count'] = data[target].apply(lambda x: len("".join(_ for _ in string.punctuation)))
            #data['title_word_count'] = data[target].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
            #data['upper_case_word_count'] = data[target].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
            self.data = data.reset_index(drop = True)
        else:
            c_ps = data.loc[data['polarity'] == 'positive'].copy()
            c_ng = data.loc[data['polarity'] == 'negative'].copy()
            c_nt = data.loc[data['polarity'] == 'neutral'].copy()

            # ---- Extract statistics of the text 
            c_ps['char_count'] = c_ps[self.target].apply(len) # Number of characters in the string
            c_ps['word_count'] = c_ps[self.target].apply(lambda x: len(x.split())) # Number of words in the string 

            c_ng['char_count'] = c_ng[self.target].apply(len) # Number of characters in the string
            c_ng['word_count'] = c_ng[self.target].apply(lambda x: len(x.split())) # Number of words in the string 

            c_nt['char_count'] = c_nt[self.target].apply(len) # Number of characters in the string
            c_nt['word_count'] = c_nt[self.target].apply(lambda x: len(x.split())) # Number of words in the string 


            c_ps = RemoveOutliers.outliers(df = c_ps, cols = ['char_count', 'word_count'])
            c_ng = RemoveOutliers.outliers(df = c_ng, cols = ['char_count', 'word_count'])
            c_nt = RemoveOutliers.outliers(df = c_nt, cols = ['char_count', 'word_count'])

            c_ps['word_density'] = c_ps['char_count'] / (c_ps['word_count']+1) # Density of word (in char)
            #c_ps['punctuation_count'] = c_ps[column].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 

            c_ng['word_density'] = c_ng['char_count'] / (c_ng['word_count']+1) # Density of word (in char)
            #c_ng['punctuation_count'] = c_ng[column].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 

            c_nt['word_density'] = c_nt['char_count'] / (c_nt['word_count']+1) # Density of word (in char)
            #c_nt['punctuation_count'] = c_nt[column].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
            data = pd.concat([c_ng,c_ps,c_nt], axis = 0,ignore_index=True)
            self.data = data


#######################################################
#               Dataset for train
#######################################################

"""
Todos os conjuntos de dados que representam um mapa de chaves para amostras de dados devem subclassificar isto.

Todas as subclasses devem sobrescrever :meth:`__getitem__`, suportando a busca de uma
amostra de dados para uma determinada chave. 

As subclasses também podem substituir opcionalmente :meth:`__len__`, que deve retornar o tamanho do conjunto
de dados por muitos
"""


class Dataset_train(Dataset):

    def __init__(self, clf, df, polarity, vocab):
        self.data = df
        self.target = np.array(polarity)
        self.vec = clf.fit_transform(self.data)#.toarray()
        self.vocab = vocab

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        
        vec = self.vec[index]
        vec = np.insert(vec, 0, self.vocab.soti["<sos>"], axis=0)
        vec = np.insert(vec, len(vec), self.vocab.soti["<eos>"], axis=0)
        
        polarity = self.target[index]

        return torch.tensor(vec, dtype= torch.float64).type(torch.LongTensor), polarity

class NumDataset_train(Dataset):
    def __init__(self, df,polarity, vocab):
        self.vecs = df
        self.targets = polarity
        
        # Start vocabulary
        self.vocab = vocab

        self.vec = None
        self.target = None

    def __len__(self):
        return self.vecs.shape[0]

    def __getitem__(self, index):
        vec = self.vecs[index]

        numericalized_caption = [self.vocab.soti["<sos>"]]
        numericalized_caption += self.vocab.numericalize(vec)
        numericalized_caption.append(self.vocab.soti["<eos>"])

        polarity = self.targets[index]

        return torch.tensor(numericalized_caption,dtype=torch.float64).type(torch.LongTensor), polarity
    
    def set_matrix(self,ds):
        
        X,y = [],[]
        size_max = 0
        for i in range(len(ds)):
            xx = ds[i][0]
            yy = ds[i][1]

            if xx.shape[0] > size_max:
                size_max = xx.shape[0]

            X.append(xx)
            y.append(yy)

        xx = []
        for i in X:
            if i.shape[0] < size_max:
                padder = torch.zeros(size_max - i.shape[0])
                padded_i = torch.cat([i,padder], dim = 0)
                xx.append(padded_i)
            else:
                xx.append(i)
        self.vec = torch.stack(xx)
        self.target = torch.tensor(y,dtype=torch.float64).type(torch.LongTensor)

#######################################################
#               Metodos de extracao
#######################################################

class extration_methods():
    
    def __init__(self,data,vocabulario,target,stopword = False):
        
        self.data  = np.array(data[target])
        self.filterp = LabelEncoder()

        self.vocab = vocabulario
        
        self.polarity = self.filterp.fit_transform(data['polarity'])

        self.stop_word = stopwords.words('portuguese') if stopword == True else None

        self.weights = self.get_weights(filters= self.filterp, polarity=self.polarity)

    @staticmethod
    def get_weights(filters,polarity):
        print('\n'+"*"*20+' COLLECTING WEIGHTS '+"*"*20, flush = True)
        class_weights = class_weight.compute_class_weight(class_weight= 'balanced',classes= np.unique(polarity),y= polarity)
        print(*[f'Class weight: {round(i[0],4)}\tclass: {i[1]}' for i in zip(class_weights, filters.classes_)], sep='\n')
        clas = filters.transform(filters.classes_)
        weights = dict(zip(clas, class_weights))

        polarity = np.array(np.unique(polarity, return_counts=True)).T
        # Determined if the dataset is balanced or imbalanced 
        ratio = np.min(polarity[:,1]) / np.max(polarity[:,1])
        if ratio > 0.1:      # Ratio 1:10 -> limite blanced / imbalanced 
            balanced = True
            print(f"\nThe dataset is balanced (ratio={round(ratio, 3)})")
        else:
            balanced = False
            print(f"\nThe dataset is imbalanced (ratio={round(ratio, 3)})")
            #from imblearn.over_sampling import ADASYN
            # put class for debalanced data 
            # in progress
        
        return weights
    

    def numericalize(self):
        ds = NumDataset_train(df= self.data, polarity = self.polarity, vocab = self.vocab)
        ds.set_matrix(ds=ds)
        return ds


    def count(self, analyzer, grams = (1,1)):

        clf = CountVectorizer(analyzer= analyzer,ngram_range = grams, stop_words = self.stop_word, vocabulary=self.vocab.soti)
        
        ds = Dataset_train(clf = clf, df = self.data, polarity = self.polarity, vocab = self.vocab)
                
        return ds
    

    def tfidf(self, analyzer, grams = (1,1)):
        
        vec = CountVectorizer(analyzer= analyzer,ngram_range = grams, stop_words = self.stop_word, vocabulary=self.vocab.soti).fit_transform(self.data)
        transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

        ds = Dataset_train(clf= transformer, df = vec, polarity=self.polarity,vocab = self.vocab)
        
        return ds
