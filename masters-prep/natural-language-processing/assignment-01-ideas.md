# Assignment 1 Ideas

Here are some ideas for things I can try out in assignment 1.

## Better Feature Extractor

### Preprocessing Techniques

1. **Tokenization**: 
    - Implement custom tokenization by splitting text on whitespace and then further splitting on punctuation marks.
    - Consider capturing emoticons as individual tokens since they can be significant for sentiment analysis.

2. **Stopword Removal**: 
    - Using a stopword list, implement a filter to remove these common words.

3. **Handling Unknown Words**: 
    - Handle unknown or rare words by either replacing them with a special `UNK` token or using a statistical method to determine whether to keep or remove them based on their frequency.
  
4. **Casing**: 
    - Lowercasing all tokens could be beneficial for uniformity unless certain words like 'not' are considered differently when capitalized.
  
5. **Stemming or Lemmatization**: 
    - Implement a simple stemming algorithm like the Porter stemmer to reduce words to their root form.
  
6. **Word Indexing**: 
    - You can construct your own index where each unique token is associated with a unique integer ID.

### Feature Extraction Techniques

1. **Bag-of-words with Weighing**: 
    - Instead of simple presence or absence, consider using a weighted bag-of-words representation.
  
2. **N-grams**: 
    - Beyond using simple words (unigrams), try using bigrams, trigrams, etc., to capture the local structure of the text.
  
3. **Feature Engineering**: 
    - Create custom features such as the number of exclamation marks, the presence of happy or sad emoticons, or any pattern you consider may be relevant for sentiment classification.
  
4. **Sentiment Lexicons**: 
    - Utilize predefined sentiment lexicons to score words in your text and use these as additional features.
  
5. **Feature Scaling**: 
    - Consider scaling features to fall within a similar range, especially if you plan to use a model sensitive to the magnitude of features.

6. **Text Length Normalization**:
    - Create a feature that represents the length of the text, and normalize all vectors by text length to counteract the variance due to text length.

7. **Handling Negation**: 
    - Identify negation words ("not", "never", etc.) and adjust the features accordingly (for instance, flipping the sign of sentiment scores for the words that follow a negation).

8. **Idf Weighing without tf-idf**: 
    - Even if you don't fully utilize tf-idf, the idf portion can be useful. You could implement this manually to emphasize rare but potentially significant words.

9. **Parts-of-Speech (POS) Tags**: 
    - You could write a simple rule-based POS tagger to identify the roles of words in a sentence and add them as features.

