import unittest
from sentiment_data import *
from utils import *
from models import UnigramFeatureExtractor, PerceptronClassifier, train_perceptron
from collections import Counter
from unittest.mock import Mock

class TestUnigramFeatureExtractor(unittest.TestCase):

    def setUp(self):
        # Initialize any required resources or mock objects
        self.indexer = Indexer()
        self.fe_extractor = UnigramFeatureExtractor(self.indexer)

    def test_init(self):
        # Test the initialization of the UnigramFeatureExtractor
        self.assertIsInstance(self.fe_extractor, UnigramFeatureExtractor)

    def test_get_indexer(self):
        # Test the get_indexer method
        retrieved_indexer = self.fe_extractor.get_indexer()
        self.assertIs(retrieved_indexer, self.indexer, "get_indexer should return the same Indexer instance")

    def test_extract_features_empty_sentence(self):
        empty_sentence = []

        # Extract features from empty sentence using the initialized feature extractor
        features = self.fe_extractor.extract_features(empty_sentence)

        # Ensure that the features extracted from an empty sentence are also empty
        self.assertEqual(len(features), 0, "Extracted features should be empty for an empty sentence")

    def test_extract_features_add_to_indexer_false(self):
        sentence = ["love"]

        # Extract features from the sentence using the initialized feature extractor
        features = self.fe_extractor.extract_features(sentence)  # add_to_indexer is False by default

        expected_features = Counter()
        self.assertEqual(features, expected_features, "Extracted features are not as expected")

        # Check if the indexer is being used correctly
        indexer = self.fe_extractor.get_indexer()
        actual_index = indexer.index_of("love")
        expected_index = -1
        self.assertEqual(actual_index, expected_index, "Indexer and expected index mismatch")


    def test_extract_features_single_word(self):
        sentence = ["love"]

        # Extract features from the sentence using the initialized feature extractor
        features = self.fe_extractor.extract_features(sentence, add_to_indexer=True)  # Adding to indexer

        # Check if the feature extraction produced the expected output
        expected_features = Counter({0: 1})
        self.assertEqual(features, expected_features, "Extracted features are not as expected")

        # Check if the indexer is being used correctly
        indexer = self.fe_extractor.get_indexer()
        actual_index = indexer.index_of("love")
        expected_index = 0
        self.assertNotEqual(actual_index, -1, "Word 'love' not found in indexer")
        self.assertEqual(actual_index, expected_index, "Indexer and expected index mismatch")

    def test_extract_features_two_words(self):
        sentence = ["love", "hate"]

        # Extract features from the sentence using the initialized feature extractor
        features = self.fe_extractor.extract_features(sentence, add_to_indexer=True)  # Adding to indexer

        # Check if the feature extraction produced the expected output
        expected_features = Counter({0:1, 1:1})
        self.assertEqual(features, expected_features, "Extracted features are not as expected")

        # Check if the indexer is being used correctly
        indexer = self.fe_extractor.get_indexer()
        love_index = indexer.index_of("love")
        self.assertNotEqual(love_index, -1, "Word 'love' not found in indexer")
        hate_index = indexer.index_of("hate")
        self.assertNotEqual(hate_index, -1, "Word 'hate' not found in indexer")

        self.assertEqual(love_index, 0, "Indexer and expected index mismatch")
        self.assertEqual(hate_index, 1, "Indexer and expected index mismatch")


class TestPerceptronClassifier(unittest.TestCase):

    def setUp(self):
        # Initialize any required resources or mock objects
        self.featurizer = Mock()  # You can create a mock featurizer here

    def test_init(self):
        # Test the initialization of the PerceptronClassifier
        perceptron = PerceptronClassifier([], self.featurizer)
        self.assertIsInstance(perceptron, PerceptronClassifier)

    def test_predict_positive(self):
        sentence = ["positive", "words"]
        weights = [1, 0]
        self.featurizer.extract_features.return_value = {0:1, 1:1}  # Define the expected feature
        perceptron = PerceptronClassifier(weights, self.featurizer)
        predicted_label = perceptron.predict(sentence)
        self.assertEqual(predicted_label, 1, "Predicted label should be 1 for positive sentiment")

    def test_predict_negative(self):
        sentence = ["negative", "words"]
        weights = [-1, 0]
        self.featurizer.extract_features.return_value = {0:1, 1:1}  # Define the expected feature
        perceptron = PerceptronClassifier(weights, self.featurizer)
        predicted_label = perceptron.predict(sentence)
        self.assertEqual(predicted_label, 0, "Predicted label should be 0 for negative sentiment")




class TestTrainPerceptron(unittest.TestCase):

    def setUp(self):
        # Initialize any required resources or mock objects
        self.indexer = Indexer()
        self.fe_extractor = UnigramFeatureExtractor(self.indexer)
        #self.train_exs = [...]  # List of SentimentExample objects for training
        # Creating a test set of SentimentExample objects
        self.train_exs = [
            SentimentExample(["i", "love", "this", "movie"], 1),
            SentimentExample(["this", "movie", "is", "terrible"], 0),
            SentimentExample(["great", "acting", "but", "boring", "plot"], 0),
            SentimentExample(["enjoyed", "the", "plot", "twist"], 1),
            SentimentExample(["didn't", "like", "the", "ending"], 0),
            SentimentExample(["amazing", "cinematography"], 1),
        ]
        # You might need to initialize other objects required for testing train_perceptron

    def test_train_perceptron(self):
        # Test the train_perceptron function
        no_train_exs = []

        # Call the train_perceptron function
        perceptron_model = train_perceptron(no_train_exs, self.fe_extractor)

        # Check if the returned model is of the correct type
        self.assertIsInstance(perceptron_model, PerceptronClassifier, "Model type is incorrect")

        # You can add more specific assertions to check the behavior of train_perceptron
        # For example, you might want to test whether the model is trained correctly on the given examples

if __name__ == '__main__':
    unittest.main()

