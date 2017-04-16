import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)


    probabilities = []
    guesses = []
    for word_id in range(test_set.num_items):
        prob_sub = {}
        
        x, length = test_set.get_item_Xlengths(word_id)
        
        best_word = None
        best_score = None

        for train_word in models.keys():
            try:
                score = models[train_word].score(x, length)
                prob_sub[train_word] = score
            except:
                prob_sub[train_word] = 0
                continue
            if best_score == None or best_score < score:
                best_score = score
                best_word = train_word

        probabilities.append(prob_sub)
        guesses.append(best_word)
    return (probabilities, guesses)
