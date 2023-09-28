from collections import Counter
from models import *
from numpy.testing import *
from sentiment_data import *
from unittest.mock import Mock
from utils import *

import logging
import numpy as np
import unittest

# Configure logging to use DEBUG level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestBigramFeatureExtractor(unittest.TestCase):

    def setUp(self):
        # Initialize any required resources or mock objects
        self.indexer = Indexer()
        self.fe_extractor = BigramFeatureExtractor(self.indexer)

    def test_init(self):
        # Test the initialization of the BigramFeatureExtractor
        self.assertIsInstance(self.fe_extractor, BigramFeatureExtractor)
    
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

    def test_extract_features_single_word(self):
        sentence = ["love"]

        # Extract features from the sentence using the initialized feature extractor
        features = self.fe_extractor.extract_features(sentence, add_to_indexer=True)  # Adding to indexer

        # Check if the feature extraction produced the expected output
        expected_features = Counter()
        self.assertEqual(features, expected_features, "Extracted features are not as expected")

        # Check if the indexer is being used correctly
        indexer = self.fe_extractor.get_indexer()
        actual_index = indexer.index_of("love")
        self.assertEqual(actual_index, -1, "Word 'love' should not have been found in indexer")

    def test_extract_features_add_to_indexer_false(self):
        sentence = ["love", "hate"]

        # Extract features from the sentence using the initialized feature extractor
        features = self.fe_extractor.extract_features(sentence)  # add_to_indexer is False by default

        expected_features = Counter()
        self.assertEqual(features, expected_features, "Extracted features are not as expected")

        # Check if the indexer is being used correctly
        indexer = self.fe_extractor.get_indexer()
        actual_index = indexer.index_of("love hate")
        expected_index = -1
        self.assertEqual(actual_index, expected_index, "Indexer and expected index mismatch")

    def test_extract_features_two_words(self):
        sentence = ["love", "hate"]

        # Extract features from the sentence using the initialized feature extractor
        features = self.fe_extractor.extract_features(sentence, add_to_indexer=True)  # Adding to indexer

        # Check if the feature extraction produced the expected output
        expected_features = Counter({0:1})
        self.assertEqual(features, expected_features, "Extracted features are not as expected")

        # Check if the indexer is being used correctly
        indexer = self.fe_extractor.get_indexer()
        love_hate_index = indexer.index_of("love hate")
        self.assertEqual(love_hate_index, 0, "bigram 'love hate' not found in indexer")
        love_index = indexer.index_of("love")
        self.assertEqual(love_index, -1, "Word 'love' should not have been found in indexer")
        hate_index = indexer.index_of("hate")
        self.assertEqual(hate_index, -1, "Word 'hate' should not have been found in indexer")

    def test_extract_features_three_words(self):
        sentence = ["love", "hate", "apathy"]

        # Extract features from the sentence using the initialized feature extractor
        features = self.fe_extractor.extract_features(sentence, add_to_indexer=True)  # Adding to indexer

        # Check if the feature extraction produced the expected output
        expected_features = Counter({0:1, 1:1})
        self.assertEqual(features, expected_features, "Extracted features are not as expected")

        # Check if the indexer is being used correctly
        indexer = self.fe_extractor.get_indexer()
        love_hate_index = indexer.index_of("love hate")
        self.assertEqual(love_hate_index, 0, "bigram 'love hate' not found in indexer")
        hate_apathy_index = indexer.index_of("hate apathy")
        self.assertEqual(hate_apathy_index, 1, "bigram 'hate apathy' not found in indexer")
        love_index = indexer.index_of("love")
        self.assertEqual(love_index, -1, "Word 'love' should not have been found in indexer")
        hate_index = indexer.index_of("hate")
        self.assertEqual(hate_index, -1, "Word 'hate' should not have been found in indexer")



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


class TestLogisticRegressionClassifier(unittest.TestCase):

    def setUp(self):
        # Initialize any required resources or mock objects
        self.featurizer = Mock()
        self.indexer = Indexer()
        self.featurizer.get_indexer.return_value = self.indexer

    def test_init(self):
        # Test the initialization of the LogisticRegressionClassifier
        classifier = LogisticRegressionClassifier([], self.featurizer)
        self.assertIsInstance(classifier, LogisticRegressionClassifier)

    def test_predict_positive(self):
        sentence = ["positive", "words"]
        weights = [1, 0]
        self.featurizer.extract_features.return_value = {0:1, 1:1}  # Define the expected feature
        self.featurizer.get_num_features.return_value = 2
        perceptron = LogisticRegressionClassifier(weights, self.featurizer)
        predicted_label = perceptron.predict(sentence, True)
        self.assertEqual(predicted_label, 1, "Predicted label should be 1 for positive sentiment")

class TestPerceptronClassifier(unittest.TestCase):

    def setUp(self):
        # Initialize any required resources or mock objects
        self.featurizer = Mock()
        self.indexer = Indexer()
        self.featurizer.get_indexer.return_value = self.indexer

    def test_init(self):
        # Test the initialization of the PerceptronClassifier
        perceptron = PerceptronClassifier([], self.featurizer)
        self.assertIsInstance(perceptron, PerceptronClassifier)

    def test_predict_positive(self):
        sentence = ["positive", "words"]
        weights = [1, 0]
        self.featurizer.extract_features.return_value = {0:1, 1:1}  # Define the expected feature
        self.featurizer.get_num_features.return_value = 2
        perceptron = PerceptronClassifier(weights, self.featurizer)
        predicted_label = perceptron.predict(sentence, True)
        self.assertEqual(predicted_label, 1, "Predicted label should be 1 for positive sentiment")

    def test_predict_negative(self):
        sentence = ["negative", "words"]
        weights = [-1, 0]
        self.featurizer.extract_features.return_value = {0:1, 1:1}  # Define the expected feature
        self.featurizer.get_num_features.return_value = 2
        perceptron = PerceptronClassifier(weights, self.featurizer)
        predicted_label = perceptron.predict(sentence, True)
        self.assertEqual(predicted_label, 0, "Predicted label should be 0 for negative sentiment")

    def test_update_weights_true_label_positive(self):
        # Create a PerceptronClassifier instance
        weights = [0, -1]
        self.featurizer.extract_features.return_value = {0:1, 1:2}  # Define the expected feature
        self.featurizer.get_num_features.return_value = 2
        classifier = PerceptronClassifier(weights, self.featurizer)

        # given the weights, the prediction should come back as 0
        prediction = classifier.predict([ "hi bye bye"], True)
        self.assertEqual(prediction, 0)

        # We set the true_label to 1 so that it doesn't match the prediction. Because the true label
        # is positive, we should increase the weights.
        true_label = 1
        alpha = 0.1

        # Update weights using the function
        is_weight_updated = classifier.update_weights(prediction, true_label, alpha)

        # Check if weights have been updated correctly
        # Weight update equation: w_new = w_old + alpha * true_label * feature_value
        #   where:
        #     true_label is +1/-1
        #     feature_value is the word count from the Counter
        # For the first feature (index 0):
        # Updated weight at index 0: 0 + 0.1 * 1 * 1 = 0.1

        # For the second feature (index 1):
        # Updated weight at index 1: -1 + 0.1 * 1 * 2 = -0.8
        expected_weights = [0.1, -0.8]
        self.assertEqual(classifier.weights, expected_weights)
        self.assertTrue(is_weight_updated)

    def test_update_weights_true_label_negative(self):
        # Create a PerceptronClassifier instance
        weights = [1, 0]
        self.featurizer.extract_features.return_value = {0:1, 1:2}  # Define the expected feature
        self.featurizer.get_num_features.return_value = 2
        classifier = PerceptronClassifier(weights, self.featurizer)

        # Initialize some example values
        prediction = classifier.predict([ "hi bye bye"], True)
        true_label = 0
        alpha = 0.1

        # Update weights using the function
        is_weight_updated = classifier.update_weights(prediction, true_label, alpha)

        # Check if weights have been updated correctly
        # Weight update equation: w_new = w_old + alpha * true_label * feature_value
        #   where:
        #     true_label is +1/-1
        #     feature_value is the word count from the Counter
        # For the first feature (index 0):
        # Updated weight at index 0: 1 + 0.1 * 1 * 1 = 1.1

        # For the second feature (index 1):
        # Updated weight at index 1: 0 + 0.1 * 1 * 2 = 0.2

        # So, the updated weights should be: [1.1, 0.2].
        updated_weights = classifier.weights
        expected_weights = [0.9, -0.2]
        self.assertEqual(updated_weights, expected_weights)
        self.assertTrue(is_weight_updated)

    def test_update_weights_no_update(self):
        # Create a PerceptronClassifier instance
        weights = [1, 0]
        classifier = PerceptronClassifier(weights, self.featurizer)

        # Initialize some example values
        features = Counter({0: 1, 1: 2})
        prediction = 1
        true_label = 1
        alpha = 0.1

        # Update weights using the function
        is_weight_updated = classifier.update_weights(prediction, true_label, alpha)

        # Ensure the weights didn't change
        updated_weights = classifier.weights
        expected_weights = [1, 0]
        self.assertEqual(updated_weights, expected_weights)
        self.assertFalse(is_weight_updated)



class TestPerceptronTrainer(unittest.TestCase):

    def setUp(self):
        self.indexer = Indexer()
        self.fe_extractor = UnigramFeatureExtractor(self.indexer)

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

    def test_no_training_examples(self):
        train_exs = []

        trainer = PerceptronTrainer(train_exs, self.fe_extractor)
        perceptron_model = trainer.train()

        self.assertIsInstance(perceptron_model, PerceptronClassifier, "Model type is incorrect")

    def test_single_training_example(self):
        train_exs = [
            SentimentExample(["i", "love", "this", "movie"], 1),
        ]

        trainer = PerceptronTrainer(train_exs, self.fe_extractor, epochs=1, seed=42, alpha=0.1)
        perceptron_model = trainer.train()

        expected_weights = np.array([0.37454 , 0.950714, 0.731994, 0.598658])

        assert_allclose(perceptron_model.weights, expected_weights, rtol=1e-6, atol=1e-6)

    def test_converging(self):
        train_exs = [
            SentimentExample(["movie", "good" ], 1),
            SentimentExample(["movie", "bad" ], 0),
            SentimentExample(["not", "good" ], 0),
        ]

        trainer = PerceptronTrainer(train_exs, self.fe_extractor, epochs=100, seed=42, alpha=0.1, shuffle=False)
        perceptron_model = trainer.train()

        # Define the expected weights based on the known seed
        expected_weights = np.array([-0.12546 ,  0.250714,  0.031994, -0.301342])

        assert_allclose(perceptron_model.weights, expected_weights, rtol=1e-6, atol=1e-6)

        self.assertTrue(perceptron_model.predict(["movie", "good" ]), "after training, the prediction should be true")
        self.assertFalse(perceptron_model.predict(["movie", "bad" ]), "after training, the prediction should be false")
        self.assertFalse(perceptron_model.predict(["not", "good" ]), "after training, the prediction should be false")

        self.assertTrue(trainer.converged, "the trainer did not converge")
        self.assertEqual(len(train_exs), trainer.number_correct_in_current_epoch, "every example must be correct if/when the model converges")


if __name__ == '__main__':
    unittest.main()
    #loader = unittest.TestLoader()
    #suite = loader.loadTestsFromName('test_models.TestTrainPerceptron.test_converging')
    #runner = unittest.TextTestRunner(verbosity=0)
    #result = runner.run(suite)

