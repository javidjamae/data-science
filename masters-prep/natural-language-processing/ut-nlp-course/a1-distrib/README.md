# Running Perceptron Training, Testing, and Unit Tests

1. **Command Line Execution**:
   - Open your terminal.
   - Navigate to the directory containing the `sentiment_classifier.py` and `test_models.py` scripts.

2. **Run Unit Tests (Optional)**:
   - Before training and testing, you can run unit tests to ensure everything is working correctly:
     ```
     python3 test_models.py
     ```
   - The unit tests validate the correctness of various components in your code.

3. **Run the Perceptron Training, Dev Evaluation, and Blind Test Prediction**:
   - To train the Perceptron model using unigram features, evaluate it on the development set, and predict on the blind test set, execute the following command:
     ```
     python3 sentiment_classifier.py --model PERCEPTRON --feats UNIGRAM
     ```
   - The script loads the training and development examples, initializes the unigram feature extractor and the Perceptron classifier, trains the model, evaluates its accuracy on the development set, and predicts on the blind test set.
   - If you want to skip predicting and writing the blind test set labels, use the `--no_run_on_test` flag:
     ```
     python3 sentiment_classifier.py --model PERCEPTRON --feats UNIGRAM --no_run_on_test
     ```
4. **Results and Outputs**:
   - The script will output accuracy, precision, recall, and F1 score for the training and development sets.
   - If not skipped, it will also write the predicted labels for the blind test set to the specified output file (default: `test-blind.output.txt`).

5. **Execution Time**:
   - The script will display the time taken for the training and evaluation process.

6. **Adjusting Parameters**:
   - If needed, you can modify the `--train_path`, `--dev_path`, `--blind_test_path`, and other arguments to point to different data files or paths.

7. **Explore Further**:
   - You can explore different feature types (`UNIGRAM`, `BIGRAM`, `BETTER`) and model types (`PERCEPTRON`, `LR`) by changing the `--feats` and `--model` arguments accordingly.
