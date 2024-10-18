from src.datasets_dataloaders import RemoveOutliers, extration_methods
from src.Vocabulary_build import Vocabulary
from src.train_func import training
from src.pdf import Metrics_LaTeX
from datetime import datetime
import pandas as pd
import argparse


def _parse_args() -> object:
    parser = argparse.ArgumentParser(
        description='cyblx trainer parser')

    parser.add_argument(
    '-ct',               '--no_classtg', 
                        action = 'store_false',
                        default = True, 
                        help="Removes outliers separately for each class")

    parser.add_argument(
    '-tg',               '--target', 
                        metavar='', 
                        type=str, 
                        default='comentario', 
                        help="Name of the column used in the tests, expected value is a string")

    parser.add_argument(
    '-ft',               '--freq_th', 
                        metavar='', 
                        type=int, 
                        default=0, 
                        help="Minimum word frequency to be considered")

    parser.add_argument(
    '-sw',               '--no_StopWord', 
                        action='store_false', 
                        default=True, 
                        help="Does not use stopwords during the analysis")

    parser.add_argument(
    '-al',               '--analyzer', 
                        metavar='', 
                        type=str, 
                        default='word', 
                        help="Choose 'word', 'char', 'char_wb', or callable to extract n-grams from text.")

    parser.add_argument(
    '-mt',               '--method', 
                        metavar='', 
                        type=str, 
                        default='tfidf', 
                        help="Method for vectorization: 'tfidf' or 'count'")

    parser.add_argument(
    '-ng',               '--Ngrams', 
                        metavar='', 
                        type=tuple, 
                        default=(1, 2), 
                        help="Range of n-grams, default value (1, 2)")

    parser.add_argument(
    '-pt',               '--proportion', 
                        metavar='', 
                        type=float, 
                        default=0.8, 
                        help="Proportion of the dataset used in the tests")

    # *************** Parametros MachineLearning ***************

    parser.add_argument(
    '-nb',               '--no_NB', 
                        action='store_false', 
                        default=True, 
                        help="Not using Naive Bayes as a test model?")

    parser.add_argument(
    '-lg',               '--no_LR', 
                        action='store_false', 
                        default=True, 
                        help="Not using Logistic Regression as a test model?")

    parser.add_argument(
    '-ss',               '--no_SVM',  
                        action='store_false', 
                        default=True, 
                        help="Not using SVM SVC as a test model?")

    parser.add_argument(
    '-kn',               '--no_KNN', 
                        action='store_false', 
                        default=True, 
                        help="Not using KNeighborsClassifier as a test model?")

    parser.add_argument(
    '-rf',               '--no_RF', 
                        action='store_false', 
                        default=True, 
                        help="Not using RandomForestClassifier as a test model?")

    parser.add_argument(
    '-sg',               '--no_SGD', 
                        action='store_false', 
                        default=True, 
                        help="Not using SGDClassifier as a test model?")

    parser.add_argument(
    '-gb',               '--no_GB', 
                        action='store_false', 
                        default=True, 
                        help="Not using GradientBoostingClassifier as a test model?")
    
    args = parser.parse_args()
    return args


def main(args):

    start_time = datetime.now()
    print("*"*20 + f"STARTING AT: {start_time} " + "*"*20, flush = True)


    path = './dataset/corpus.csv'

    refs = pd.read_csv(path).dropna().drop(columns = ['Unnamed: 0'])
    refs = refs.sample(frac=args.proportion).reset_index(drop = True)
    refs = refs.astype({'comentario' : 'string',
                'tokens_stem' : 'string',
                'tokens_lemma' : 'string',
                'simple_POS': 'string',
                'detailed_POS': 'string',
                'Syntactic_dependency': 'string',
                'polarity':'string'})


    print('\n'+"*"*10 + " Remove Outliers " + "*"*10, flush = True)
    RO = RemoveOutliers(ClassTogether = args.no_classtg, target = args.target)
    RO.statistics(data = refs)

    VB = Vocabulary(target = args.target, freq_threshold = args.freq_th,stopword = args.no_StopWord)
    VB.extract_vocab(data = RO.data) #Incluir removacao de stopwords

    methods = extration_methods(data = RO.data,vocabulario = VB,target = args.target, stopword= args.no_StopWord)
    
    if args.method == 'tfidf':
        dataset = methods.tfidf(analyzer = args.analyzer,grams = args.Ngrams )
    elif args.method == 'count':
        dataset = methods.count(analyzer = args.analyzer,grams = args.Ngrams )
    else:
        raise ValueError("Invalid method specified. Use 'tfidf' or 'count'.")

    tt = training(args, labels = methods.filterp, weights= methods.weights)

    model_ = tt.learning(args,
                         ds = dataset,
                         bd_f = args.proportion, proportion= 0.8)
    tt.save_report('./report')

    pdf = Metrics_LaTeX()
    pdf.Read_Generate()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
# %%
