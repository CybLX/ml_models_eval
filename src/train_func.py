from sklearn import naive_bayes,linear_model,svm,ensemble,neighbors
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from src.class_metric import Metrics
import torch
import json
import os

class training:
    def __init__(self,args,labels, weights):
        self.labels = labels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights = weights
        self.metrics = Metrics()
        self.report = {"proportion"        : args.proportion,
                        "analyzer"         : args.analyzer,
                        "method"           : args.method,
                        "Ngrams"           : args.Ngrams,
                        "freq_threshold"   : args.freq_th,
                        "stop_word"        : args.no_StopWord,
                        "class_together"   : args.no_classtg}
        
    @staticmethod
    def RandomSplit(ds,proportion=0.8):
        # Random split of 80:20 between training and validation
        num_items = len(ds)
        num_train = round(num_items * proportion)
        num_val = num_items - num_train
        train_ds, val_ds = random_split(ds, [num_train, num_val]) #separa as amostras

        return train_ds,val_ds

    def save_report(self,path):
        if "metrics.json" not in os.listdir(path):
            with open(path + '/metrics.json','w') as f:
                f.write(json.dumps(self.report))
                f.close()
            print('Report Created!')
        else:
            with open(os.path.join(path + "/metrics.json"), "r+") as jsonFile:
                data = json.load(jsonFile)
                jsonFile.truncate(0)
                jsonFile.seek(0)
                data.update(self.report)
                json.dump(data, jsonFile, indent=4)
                jsonFile.close()
            print('Report Updated!')

    def learning(self,args, ds, bd_f,proportion = 0.8):
        X = ds.vec
        y = ds.target
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=proportion)

        print('\n'+"*"*20 + " START TRAINING " + "*"*20, flush = True)

        if args.no_NB:
            print("\nMultinomial Naive Bayes\n", flush = True)
            path = 'Naive_Bayes'
            model_, time_train, score_, class_rp, kappa_, lr_auc, f1_auc = self.metrics.metrics_ML(naive_bayes.MultinomialNB(), X_train, y_train, X_val, y_val,labels = self.labels.classes_ , model_name = path, gb=True)
            
            self.report[path] = {}
            self.report[path]['BigData_fraction'] = bd_f
            self.report[path]['time_train']   = time_train
            self.report[path]['score_']       = score_
            self.report[path]['class_rp']     = class_rp
            self.report[path]['kappa_']       = kappa_
            self.report[path]['lr_auc']       = lr_auc
            self.report[path]['f1_auc']       = f1_auc
            print("Finish\n", flush = True)

        if args.no_LR:
            print("\nLogistic Regression\n", flush = True)
            path = 'Logistic_Regression'
            model_, time_train, score_, class_rp, kappa_, lr_auc, f1_auc = self.metrics.metrics_ML(
                                                                        linear_model.LogisticRegression(class_weight= self.weights,max_iter=1000,  random_state=42),
                                                                        X_train, y_train, X_val, y_val, labels = self.labels.classes_, model_name = path, gb=True)
            
            self.report[path] = {}
            self.report[path]['BigData_fraction'] = bd_f
            self.report[path]['time_train']   = time_train
            self.report[path]['score_']       = score_
            self.report[path]['class_rp']     = class_rp
            self.report[path]['kappa_']       = kappa_
            self.report[path]['lr_auc']       = lr_auc
            self.report[path]['f1_auc']       = f1_auc
            print("Finish\n", flush = True)

        if args.no_SVM:
            print("\nSVM\n", flush = True)
            path = 'Svm_Svc'
            model_, time_train, score_, class_rp, kappa_, lr_auc, f1_auc = self.metrics.metrics_ML(svm.SVC(class_weight= self.weights, probability= True),  X_train, y_train, X_val, y_val,labels = self.labels.classes_, model_name = path, gb=True)
            
            self.report[path] = {}
            self.report[path]['BigData_fraction'] = bd_f
            self.report[path]['time_train']   = time_train
            self.report[path]['score_']       = score_
            self.report[path]['class_rp']     = class_rp
            self.report[path]['kappa_']       = kappa_
            self.report[path]['lr_auc']       = lr_auc
            self.report[path]['f1_auc']       = f1_auc
            print("Finish\n", flush = True)

        if args.no_KNN:
            print("\nk-NN\n", flush = True)
            path = 'Kneighbors_Classifier'
            model_, time_train, score_, class_rp, kappa_, lr_auc, f1_auc = self.metrics.metrics_ML(neighbors.KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1),  X_train, y_train, X_val, y_val,labels = self.labels.classes_ , model_name = path, gb=True)
            
            self.report[path] = {}
            self.report[path]['BigData_fraction'] = bd_f
            self.report[path]['time_train']   = time_train
            self.report[path]['score_']       = score_
            self.report[path]['class_rp']     = class_rp
            self.report[path]['kappa_']       = kappa_
            self.report[path]['lr_auc']       = lr_auc
            self.report[path]['f1_auc']       = f1_auc
            print("Finish\n", flush = True)

        if args.no_RF:
            print("\nRandom Forest\n", flush = True)
            path = 'Random_Forest'
            model = ensemble.RandomForestClassifier(class_weight=self.weights)
            model_, time_train, score_, class_rp, kappa_, lr_auc, f1_auc = self.metrics.metrics_ML(model,  X_train, y_train, X_val, y_val,labels = self.labels.classes_ , model_name = path, gb=True)
            
            self.report[path] = {}
            self.report[path]['BigData_fraction'] = bd_f
            self.report[path]['time_train']   = time_train
            self.report[path]['score_']       = score_
            self.report[path]['class_rp']     = class_rp
            self.report[path]['kappa_']       = kappa_
            self.report[path]['lr_auc']       = lr_auc
            self.report[path]['f1_auc']       = f1_auc
            print("Finish\n", flush = True)
            
        if args.no_SGD:
            print("\nStochastic Gradient Descent with early stopping\n", flush = True)
            print("Early Stopping : 100 iterations without change", flush = True)
            path = 'SGD_Classifier'
            model = linear_model.SGDClassifier(class_weight= self.weights,loss='modified_huber', max_iter=1000, tol=1e-3,   n_iter_no_change=1000, early_stopping=True, n_jobs=-1 )
            model_, time_train, score_, class_rp, kappa_, lr_auc, f1_auc = self.metrics.metrics_ML(model,  X_train, y_train, X_val, y_val,labels = self.labels.classes_ , model_name = path, gb=True)
            
            self.report[path] = {}
            self.report[path]['BigData_fraction'] = bd_f
            self.report[path]['time_train']   = time_train
            self.report[path]['score_']       = score_
            self.report[path]['class_rp']     = class_rp
            self.report[path]['kappa_']       = kappa_
            self.report[path]['lr_auc']       = lr_auc
            self.report[path]['f1_auc']       = f1_auc
            print("Finish\n", flush = True)
            
        if args.no_GB:
            print("\nGradient Boosting with early stopping\n", flush = True)
            print("Early Stopping : 100 iterations without change", flush = True)
            path = 'Gradient_Boosting'
            model_, time_train, score_, class_rp, kappa_, lr_auc, f1_auc = self.metrics.metrics_ML(ensemble.GradientBoostingClassifier(n_estimators=100, validation_fraction=0.2,n_iter_no_change=1000, tol=1e-3,random_state=0, verbose=0), X_train, y_train, X_val, y_val,labels = self.labels.classes_ , model_name = path, gb=True)
            
            self.report[path] = {}
            self.report[path]['BigData_fraction'] = bd_f
            self.report[path]['time_train']   = time_train
            self.report[path]['score_']       = score_
            self.report[path]['class_rp']     = class_rp
            self.report[path]['kappa_']       = kappa_
            self.report[path]['lr_auc']       = lr_auc
            self.report[path]['f1_auc']       = f1_auc
            print("Finish\n", flush = True)

        print('\n'+"*"*20 + "Done!" + "*"*20, flush = True)
