from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time

class Metrics(object):
    '''
    Class containing functions to plot the different metric curves (Precision-Recall, ROC AUC etc...)
    '''
    
    def __init__(self):
        '''
        Initialisation of the class'''
        

    @classmethod 
    def roc_auc_curve(self, model, x, y, labels, model_name, gb=False, deep = False):
        '''
        Function to plot the ROC AUC curves for binary or multiclass classification. 
        Correct for standard machine learning models and Neural Networks. 
        @param model: (model) classification model
        @param x: (list) validation sample
        @param y: (list int) validation sample label
        @param gb: (bool) inform if the model is an ensemble model 
        '''
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(y.reshape(-1, 1)))]
        
        # predict
        if deep:
            lr_probs = model(x).detach().numpy()
            print(lr_probs)
        else:
            # predict probabilities
            if gb: # test if the model is an ensemble model 
                lr_probs = model.predict_proba(x)
            else:
                lr_probs = model.predict(x)
        

        plt.figure()

        dummy_y = self.to_categorical(y)
        lr_auc_multi = []
        for i in enumerate(labels):
            lr_auc_multi.append(round(roc_auc_score(dummy_y[:,i[0]], lr_probs[:,i[0]], average="weighted"),3))
            print(f"ROC AUC class {i[1]}: {lr_auc_multi[-1]}")
        print(lr_auc_multi)
        
        lr_auc = roc_auc_score(dummy_y, lr_probs, average="weighted", multi_class="ovr" )
        ns_fpr, ns_tpr = [i/10 for i in list(range(0, 11, 1))], [i/10 for i in list(range(0, 11, 1))] #f(range(0, 0.1, 1))
        
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        for i in range(lr_probs.shape[1]):
            lr_fpr, lr_tpr, _ = roc_curve(dummy_y[:,i], lr_probs[:,i])
            # plot the roc curve for the model
            plt.plot(lr_fpr, lr_tpr, label=f'Class {labels[i]} (area {lr_auc_multi[i]})')
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # show the grid
            plt.grid(True)
            # show the legend
            plt.legend()
            
        print('\nROC AUC=%.3f \n' % (lr_auc))
        plt.savefig(f'./report/Learning/{model_name}/True-Positive-Rate.png')
        #plt.show()
        return lr_auc

    @classmethod 
    def confusion_matrix(self, model, y, x, labels):
        '''
        Compute the confusion matrix for binary or multiclass classification. 
        Correct for standard machine learning models and Neural Networks. 
        @param model: (model) classification model
        @param x: (list) validation sample
        @param y: (list int) validation sample label
        
        '''
        if len(labels)==2: # binary confusion matrix
            confu_matrix = pd.DataFrame(confusion_matrix(y, (model.predict(x) > 0.5).astype(int)), \
                 columns=['Predicted Negative', "Predicted Positive"], index=['Actual Negative', 'Actual Positive'])
            print(confu_matrix)
            return confu_matrix
        else:
            # multiclass confusion matrix
            dummy_y = self.to_categorical(y)
            mcm = multilabel_confusion_matrix(dummy_y, self.to_categorical(model.predict(x).argmax(-1)))
            df_mcm = pd.DataFrame()
            for i in zip(mcm, labels): # compute confusion matrix for each class 
                mcm = pd.DataFrame(data=i[0], columns=['Predicted Negative', "Predicted Positive"], index=['Actual Negative', 'Actual Positive'])
                df_mcm = df_mcm.append(mcm)
                print("\nConfusion matrix for classe: %s \n" %(i[1]))
                print(mcm)
                print("\n")
            return df_mcm
        
    @classmethod 
    def precision_recall_curve(self, model, x, y, labels, model_name, gb=False):
        '''
        Function to plot the recall precision curves for binary or multiclass classification. 
        Correct for standard machine learning models and Neural Networks. 
        @param model: (model) classification model
        @param x: (list) validation sample
        @param y: (list int) validation sample label
        @param gb: (bool) inform if the model is an ensemble model 
        '''
        if gb:# test if the model is an ensemble model 
            # predict probabilities
            lr_probs = model.predict_proba(x)
        else:
            lr_probs = model.predict(x)
        

        print("\n")
        plt.figure(dpi = 300)     
        indexs = [] 
        f1 = []
        aucs = []

        dummy_y = self.to_categorical(y)
        dummy_lr = self.to_categorical(lr_probs.argmax(-1))
        for i in enumerate(labels):
            precision, recall, thresholds = precision_recall_curve(dummy_y[:,i[0]], lr_probs[:,i[0]])
            # calculate precision-recall AUC
            lr_f1 = f1_score(dummy_y[:,i[0]], dummy_lr[:,i[0]]) 
            lr_auc = auc(recall, precision)
            # summarize scores
            print('Model class: %s --> f1-score=%.3f AUC=%.3f' % (i[1], lr_f1, lr_auc))
            indexs.append(i[1])
            f1.append(lr_f1)
            aucs.append(lr_auc)
            plt.plot(recall, precision, label='Class %s' %(i[1]))
        
        no_skill = len(y[y>=1]) / len(y)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        # plot the precision-recall curves
        print("\n")

        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the grid
        plt.grid(True)
        # show the plot
        plt.savefig(f'./report/Learning/{model_name}/Recall-Precision.png')
        return (indexs,f1,aucs)

    @classmethod 
    def plot_eval_xgb(self, model, labels):
        '''
        Function to plot the evaluation curves for xgboost models 
        @param model: (model) xgboost model
        @param labels: (list) list ocntaining the labels in string 
        '''
        # retrieve performance metrics
        results = model.evals_result()
        if len(labels)>2: # multiclass 
            log_ = "mlogloss"
            error_= "merror"
        else: # binary classifiation
            log_ = "logloss"
            error_= "error"

        # create axis x with the number of epochs
        epochs = len(results['validation_0'][error_])
        x_axis = range(0, epochs)

        plt.figure(figsize=(15,10))
        plt.subplot(221)
        # Plot training & validation accuracy values
        plt.plot(x_axis, results['validation_0'][log_], label='Train')
        plt.plot(x_axis, results['validation_1'][log_], label='Test')
        plt.ylabel('Log Loss')
        plt.xlabel('Epochs')
        plt.title('XGBoost Log Loss')
        plt.legend(loc='upper left')
        plt.grid(True)


        # Plot training & validation loss values
        plt.subplot(222)
        plt.plot(x_axis, results['validation_0'][error_], label='Train')
        plt.plot(x_axis, results['validation_1'][error_], label='Test')
        plt.legend()
        plt.ylabel('Classification Error')
        plt.xlabel('Epochs')
        plt.title('XGBoost Classification Error')
        plt.legend( loc='upper left')
        plt.grid(True)
        #plt.show()
        
    @classmethod 
    def plot_confusion_matrix(self, cm, classes,model_name ,normalized=True, cmap='bone'):
        '''
        Function to generate an heatmap of the confusion matrix
        @param cm: (matrix) confusion matrix
        @param classes: (list) list containing labels of the classes
        @param normalised: (bool) determined if the confusion matrix need to be normalized
        @param cmap: (str) color for the confusion matrix
        '''
        plt.figure()
        norm_cm = cm
        if normalized:
            norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
            plt.savefig(f'./report/Learning/{model_name}/confusion-matrix.png')

    @classmethod
    def plot_history(self, history):
        '''
        Function to plot the learning curves of a neural network
        @param history: metrics of a neural network
        '''
        plt.figure(figsize=(15,10))
        plt.subplot(221)
        # Plot training & validation accuracy values
        plt.plot(history['train_accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.grid(True)


        # Plot training & validation loss values
        plt.subplot(222)
        plt.plot(history['train_loss'])
        plt.plot(history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.grid(True)
        #plt.show()
            

    @classmethod
    def metrics_ML(self,clf, X, Y, X_test, Y_test, labels, model_name, gb=False):
        '''
        Function to compute a classifier model
        @param X: (matrix) training x data
        @param y: (list) train labels - ground truth
        @param x_test: (matrix) matrix of test x data
        @param y_test: (list) list of test labels
        
        @return model_NB: (model) train model
        @return pred: (list) list of predicted labels
        @return end: (float) fit time of the model
        @return model.score(): (float) accuracy of the model
        '''
        model_, pred, time_train, score_ = self.classifier_model(clf, X, Y, X_test, Y_test)
        print("Execution time : %.3f s" %(time_train))

        
        score_ = round(100*score_,2)
        print(f"Score : {score_} %" )
        print("\nClassification Report\n")
        class_rp = classification_report(Y_test, pred, target_names=labels, output_dict=True)
        print(class_rp)
        cm = confusion_matrix(Y_test, pred)
        print("\nConfusion Matrix\n")
        self.plot_confusion_matrix(cm, labels, model_name=model_name)
        print("\n")
        f1_auc = self.precision_recall_curve(model_,  X_test, Y_test, labels, model_name=model_name, gb=gb)
        print("\n")
        lr_auc = self.roc_auc_curve(model_,  X_test, Y_test, labels, model_name=model_name, gb=gb)
        kappa_ = round(100*cohen_kappa_score(Y_test,  pred),2)
        print(f"\n\nCohen's kappa: {kappa_}%\n\n")

        return model_, time_train, score_, class_rp, kappa_, lr_auc, f1_auc

    @staticmethod
    def classifier_model(clf, X, y, x_test, y_test):
        '''
        Function to compute a classifier model
        @param clf: (model) classifier model
        @param X: (matrix) training x data
        @param y: (list) train labels - ground truth
        @param x_test: (matrix) matrix of test x data
        @param y_test: (list) list of test labels
        @return model_NB: (model) train model
        @return pred: (list) list of predicted labels
        @return end: (float) fit time of the model
        @return model.score(): (float) accuracy of the model
        '''

        start = time.time()
        clf.fit(X, y)
        end = time.time() - start
        pred = clf.predict(x_test)


        return clf, pred, end, clf.score(x_test, y_test)
    
    @staticmethod
    def to_categorical(y, num_classes=None, dtype="float32"):
        """Converts a class vector (integers) to binary class matrix.

        E.g. for use with `categorical_crossentropy`.

        Args:
            y: Array-like with class values to be converted into a matrix
                (integers from 0 to `num_classes - 1`).
            num_classes: Total number of classes. If `None`, this would be inferred
              as `max(y) + 1`.
            dtype: The data type expected by the input. Default: `'float32'`.

        Returns:
            A binary matrix representation of the input. The class axis is placed
            last.

        Example:

        >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
        >>> a = tf.constant(a, shape=[4, 4])
        >>> print(a)
        tf.Tensor(
          [[1. 0. 0. 0.]
           [0. 1. 0. 0.]
           [0. 0. 1. 0.]
           [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)

        >>> b = tf.constant([.9, .04, .03, .03,
        ...                  .3, .45, .15, .13,
        ...                  .04, .01, .94, .05,
        ...                  .12, .21, .5, .17],
        ...                 shape=[4, 4])
        >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
        >>> print(np.around(loss, 5))
        [0.10536 0.82807 0.1011  1.77196]

        >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
        >>> print(np.around(loss, 5))
        [0. 0. 0. 0.]
        """
        y = np.array(y, dtype="int")
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical
