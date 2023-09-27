# models.py
from collections import Counter
from sentiment_data import *
from utils import *
import logging
import numpy as np
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import nltk
    nltk.download('stopwords')
    logger.info("Downloaded stopwords from nltk...")
except ImportError:
    logger.error("nltk not available. Couldn't download the stopword list.")
    
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    logger.info("Loaded stopwords from nltk...")
except ImportError:
    logger.error("nltk.corpus.stopwords not available. Using a fallback list.")
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
    logger.info("Using fallback stopwords list...")

# Define a module-level variable for the random seed
RANDOM_SEED = None

def set_random_seed(seed):
    global RANDOM_SEED
    RANDOM_SEED = seed
    np.random.seed(RANDOM_SEED)


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        The indexer assigns unique indices to words, ensuring consistent numeric representation.
        The Counter stores word counts for the current sentence using index/count pairs. For example,
            if we extract "hi Javid hi", hi would get indexed at 0 and Javid would get indexed at 1,
            and we'd get a Counter with { 0:2, 1:1 }. If we call it again with "bye Javid bye Javid",
            it would index bye as 2 and we'd get { 1:2, 2,2 }.
        """
        logger.debug(f"UnigramFeatureExtractor::extract_features -- sentence: {sentence}")
        counter = Counter()
        for word in sentence:
            index = self.indexer.add_and_get_index( word, add=add_to_indexer )
            if index >= 0: # Only consider words that are present in the indexer
                counter[index] += 1

        logger.debug(f"UnigramFeatureExtractor::extract_features -- counter: {counter}")
        return counter


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        The indexer assigns unique indices to words, ensuring consistent numeric representation.
        The Counter stores word counts for the current sentence using index/count pairs. For example,
            if we extract "hi Javid hi", hi would get indexed at 0 and Javid would get indexed at 1,
            and we'd get a Counter with { 0:2, 1:1 }. If we call it again with "bye Javid bye Javid",
            it would index bye as 2 and we'd get { 1:2, 2,2 }.
        """
        logger.debug(f"BigramFeatureExtractor::extract_features -- sentence: {sentence}")
        counter = Counter()
        for i in range(len(sentence) - 1):
            index = self.indexer.add_and_get_index( sentence[i] + " " + sentence[i+1], add=add_to_indexer )
            if index >= 0: # Only consider words that are present in the indexer
                counter[index] += 1

        logger.debug(f"BigramFeatureExtractor::extract_features -- counter: {counter}")
        return counter



class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: List[float], featurizer: FeatureExtractor):
        self.weights = weights
        self.featurizer = featurizer
        self.prediction_features = None

    def predict(self, sentence: List[str], is_training: bool = False) -> int:
        # Extract features using the featurizer, which will return the Counter
        logger.debug(f"PerceptronClassifier::predict -- sentence: {sentence}, seed: {RANDOM_SEED}")

        self.prediction_features = self.featurizer.extract_features(sentence, is_training)
        logger.debug(f"PerceptronClassifier::predict -- prediction_features: {self.prediction_features}, count: {len(self.prediction_features)}")

        num_features = len(self.featurizer.get_indexer())

        # Update the size of the weights array if needed
        if num_features > len(self.weights):
            additional_weights = np.random.random(num_features - len(self.weights))
            self.weights = np.concatenate((self.weights, additional_weights))

        # Perform the prediction based on features and weights
        score = sum(self.weights[index] * value for index, value in self.prediction_features.items())
        if score >= 0:
            predicted_label = 1  # Positive class
        else:
            predicted_label = 0  # Negative class

        logger.debug(f"PerceptronClassifier::predict -- num_features: {num_features}, Weights: {self.weights}, Score: {score}, Predicted Label: {predicted_label}")

        return predicted_label

    def update_weights(self, prediction: int, true_label: int, alpha: float) -> bool:
        """
        Update the weights of the Perceptron based on the prediction and true label.
        :param features: Dictionary of features extracted from the input data
        :param prediction: The predicted label (0 or 1)
        :param true_label: The true label (0 or 1)
        :param alpha: Learning rate
        :return: true if the weights were updated, false if they were not
        """
        logger.debug(f"PerceptronClassifier::update_weights - Predicted Label: {prediction}, True Label: {true_label}, Alpha: {alpha}")

        if prediction != true_label:
            adjusted_true_label = 2 * true_label - 1  # Map 0 to -1 and 1 to +1
            logger.debug(f"PerceptronClassifier::update_weights - Adjusted True Label: {adjusted_true_label}")

            for index, value in self.prediction_features.items():
                # value is the word count
                self.weights[index] += alpha * adjusted_true_label * value
            return True
        else:
            return False

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self):
        raise Exception("Must be implemented")

class LearningRateSchedule():
    def get_new_rate( epoch:int ):
        raise( "instantiate this in the subclass" )

class FixedLearningRateSchedule( LearningRateSchedule ):
    def __init__( self, initial_rate:float = 1 ):
        self.rate = initial_rate

    def get_new_rate( self, epoch:int ):
        return self.rate

class OneOverTeeLearningRateSchedule(LearningRateSchedule):
    def __init__( self, initial_rate:float = 1 ):
        self.rate = initial_rate

    def get_new_rate( self, epoch:int ):
        if ( epoch > 0 ):
            self.rate = 1 / epoch
        return self.rate

class PerceptronTrainer():
    def __init__(self, train_exs:List[SentimentExample], feat_extractor:FeatureExtractor, epochs:int=10, seed:int=None, alpha:float=0.1, shuffle:bool=True, learning_rate_schedule:LearningRateSchedule=None):
        self.train_exs = train_exs
        self.feat_extractor = feat_extractor
        self.epochs = epochs
        self.alpha = alpha
        self.number_correct_in_current_epoch = 0
        self.weights = np.array([])  # Initialize weights with an empty array
        self.classifier = PerceptronClassifier( self.weights, self.feat_extractor )
        self.converged = False
        self.shuffle = shuffle

        if seed is not None:
            set_random_seed(seed)

        if learning_rate_schedule is not None:
            self.learning_rate_schedule = learning_rate_schedule
        else: # no schedule provided, keep it the same across epochs
            self.learning_rate_schedule = FixedLearningRateSchedule(alpha)

    def train(self):
        for t in range( 1, self.epochs + 1 ):
            logger.debug("--------------------------------------")
            logger.debug(f"PerceptronTrainer.train:: epoch: {t}")
            self.number_correct_in_current_epoch = 0
            self.alpha = self.learning_rate_schedule.get_new_rate( t )

            if( self.shuffle ):
                random.shuffle(self.train_exs)

            for index, example in enumerate(self.train_exs):
                prediction = self.classifier.predict( example.words, True )
                is_weight_updated = self.classifier.update_weights( prediction, example.label, self.alpha )
                logger.debug(f"PerceptronTrainer.train:: Index: {index}, Example: {example.words}, Label: {example.label}, Prediction: {prediction}, Updated Weights: {self.classifier.weights}, is_weight_updated: {is_weight_updated}")
                if( not is_weight_updated ):
                    self.number_correct_in_current_epoch += 1

            if ( self.number_correct_in_current_epoch == len( self.train_exs ) ):
                logger.debug(f"PerceptronTrainer.train:: CONVERGED!! Exiting on epoch {t}")
                self.converged = True
                break
        return self.classifier

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, epochs: int = 40, seed: int = None, alpha: float = 0.1) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :param epochs: number of training epochs (optional)
    :param seed: random seed for reproducibility (optional)
    :return: trained PerceptronClassifier model
    """
    learning_rate_schedule = OneOverTeeLearningRateSchedule(1)
    #learning_rate_schedule = None
    return PerceptronTrainer(train_exs, feat_extractor, epochs=epochs, seed=seed, alpha=alpha, shuffle=True, learning_rate_schedule=learning_rate_schedule).train()


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    raise Exception("Must be implemented")


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
