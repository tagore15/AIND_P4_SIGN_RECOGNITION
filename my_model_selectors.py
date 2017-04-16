import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ''' 
       Number of parameters are calculated as per below calculation expalined by Udacity mentor in slack
        p = n*(n-1) + (n-1) + 2*d*n = n^2 + 2*d*n - 1 where d is number of features
        There is one thing a little different for our project though... in the paper, the initial distribution is estimated and therefore those parameters are not "free parameters".  However, hmmlearn will "learn" these for us if not provided.  Therefore they are also free parameters:
        => p = n*(n-1) + (n-1) + 2*d*n
               = n^2 + 2*d*n - 1
        '''
        
        bestScore = None
        bestModel = None

        for num in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num)
            
			    # number of parameters
                p = num*(num-1) + (num-1) + 2*len(self.sequences[0][0]) * num

                # number of data points
                N = len(self.sequences)

                score = -2*model.score(self.X, self.lengths) + p * np.log(N)
            
                if bestScore == None or score < bestScore:
                    bestScore = score
                    bestModel = model
            except:
                continue
        
        return bestModel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        bestScore = None
        bestModel = None
        
        for num in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num)
                logScore = model.score(self.X, self.lengths) 

                otherScore = 0
                for word in self.words:
                    if word != self.this_word:
                        x, length = self.hwords[word]
                        otherScore += model.score(x, length) 
                    
                score = logScore - (otherScore/(len(self.hwords)-1))

                if bestScore == None or score > bestScore:
                    bestScore = score
                    bestModel = model 
            except:
                continue

        return bestModel 


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        bestScore = None
        bestModel = None
        if (len(self.sequences) == 1): 
            # run without splitting
            for num in range(self.min_n_components, self.max_n_components+1):
                try:
                    model = self.base_model(num)
                    score = model.score(self.X, self.lengths) 

                    if bestScore == None or score > bestScore:
                        bestScore = score
                        bestModel = model 
                except:
                    continue
                 
            return bestModel 
        
        # run with cv splitting
        num_splits = min(len(self.sequences), 3)
        kf = KFold(n_splits=num_splits)
        for num in range(self.min_n_components, self.max_n_components+1):
            sum_score = 0
            model = None
            for train_index, test_index in kf.split(self.sequences):
                X_train, lengths_train = combine_sequences(train_index, self.sequences)
                try:
                    model = GaussianHMM(n_components=num, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False)
                    model.fit(X_train, lengths_train)

                    X_test, lengths_test = combine_sequences(test_index, self.sequences)
                    score_test = model.score(X_test, lengths_test)
                    sum_score += score_test
                except:
                    continue

            avg_score = sum_score/num_splits 
            if bestScore == None or avg_score > bestScore:
                bestScore = avg_score
                bestModel = model
        return bestModel
