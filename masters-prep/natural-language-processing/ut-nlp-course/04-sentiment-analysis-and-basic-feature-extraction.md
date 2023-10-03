# Sentiment Analysis and Basic Feature Extraction

These are my notes from [this lecture](https://www.youtube.com/watch?v=0jSElGFUxro&list=PLofp2YXfp7TZZ5c7HEChs0_wfEfewLDs7&index=4) that is part of [this course](https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html).

## Lecture Notes

In this lecture we'll discuss doing classification for sentiment analysis. This will help us understand how to go from text to feature vectors, which we can then use in a classifier where we take training data and train a classification model.

Example

> the movie was great! would watch again! [+]

This is a positive sentiment given the `[+]` label. 

Why would we think it's positive? The use of the words `great` and `watch again` help us draw this conclusion.

Example
> the film was awful; I'll never watch again! [-]

Here we have the word `awful` but we still have `watch again`, but it is negated with `never`.

Sentiment analysis is a complicated task because of factors like negation and other higher-level structures.

The basic approach we'll take to solve this problem is using a "Bag of Words".

Process that we'll take
1. **Feature Extraction** - Turn text into features

$$\overline{x} \Rightarrow f(\overline{x})$$

2. **Classifier training** - We use labeled training data to train the model

$$
\left\{ f(\overline{x}^{[i]}), y^{[i]} \right\} \vphantom{\sum_{i=1}^{D}} _{i=1}^D
$$

Where
- D is the size of the dataset of labeled examples


### Feature Extraction

How do we turn an example into a feature vector?

Example
> the movie was great

**Bag-of-words**: Assume we have 10,000 words in our vocabulary. For example, the 10K most common words in English.

We lay them out in a big vector:

```[ the a of at ... movie ... was ... great ... ]```

We can create a sparse vector for each example where we'd have `1` for each position where we have a word and `0` everywhere else. So for the above example, we'd have 4 `1`'s and 9,996 `0`'s.

The values can either be counts representing how many times each word is present, or it could just represent 1 or 0 for presence or absence, respectively. 

**Bag-of-ngrams**
An **n-gram** is a sequence of n-consecutive words.

2-grams: (the movie), (movie was), (was great)

**tf-idf**
We won't use this too much in the course. 

Term Frequency (tf): the count of the term in the example. 

Inverse Document Frequency (idf)

$$
\log\frac{N}{\left\{ D: w\epsilon D \right\}}
$$

Where:
* $N$: Total number of documents
* $\left\{ D: w\epsilon D \right\}$: documents with $w$ in them

We'd use tf-idf? If we have `the` in many documents, so the log term will be close to 0. The tf-idf is the term frequency times the inverse document frequency. So `the` is going to receive a value that is close to 0. But if you have a word that shows up in a single document, but is rare in other documents, then the log term will be large. So it emphasizes characteristic words from the given document. 

### Preprocessing

#### Tokenization
Tokenization is a process of splitting words.

Example:
```
was great
was great!
```

If we just used white-space tokenization we'd have two words for great (`great` and `great!`).

So in our vocabulary we'd have:

```
[ ... great ... great! ... ]
```

So a tokenizer will take a string and turn it into a string with additional spaces. 

```
was great !
```

Another example:

```
wasn't
```

can turn into:

```
was n't
```

#### Stopword Removal
Sometimes we want to remove common function words that don't add too much value (the, of, a, etc).

#### Casing
Sometimes we want to lowercase or truecase the data

#### Handling Unknown Words
If you have a rare word like "Durrett", it won't be in the vocabulary, so we might replace it with an `UNK` token, or drop it (common in sentiment analysis).

#### Indexing
Mapping each word (or n-gram, etc) into the space of N (natural numbers). Basically, use a map to store where each token is in our vocabulary. 

