# Optimization Basics

These are my notes from [this lecture](https://www.youtube.com/watch?v=8EqQROdVPyM&list=PLofp2YXfp7TZZ5c7HEChs0_wfEfewLDs7&index=20) that is part of [this course](https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html).

## Lecture Notes

Word embeddings are one of the most important concepts in NLP to emerge over the last 10 years (since around 2013 timeframe). They allow us to bridge the gap from raw text to representations that can work well for Neural Networks. 

Neural Networks have been around a long time and they are really good at learning functions over continuous data, but for a long time it wasn't clear how to take discrete data (e.g. words, bag-of-words representations) and get them to work well with neural nets. Word Embeddings are the first step towards bridging that gap, and a key piece to enable the neural revolution in NLP.

### Word Embeddings

Example:
```
movie was good -> [ 0 1 0 0 0 1 0 0 0 0 ... 1 0 0 ]
```

The ones in that vector would represent the words in the vocabulary. Let's just assume the first `1` represents `was`, the second represents `good` and the third represents `movie`.

We can represent this as three separate vectors:
```
Movie
[ 0 ... 1 0 0 ]

Was
[ 0 1 0 ... ]

Good
[ 0 ... 1 ... 0 ] 
```

Now we can sum them together to get our bag-of-words representation. 

If we then have:
```
film is great
```

We'd have an orthogonol set to `movie was good`. There is no apparent connection, no overlapping words, so the dot product would be 0. If we have a lot of data, we'd learn appropriate weights for all of these. 

But, this shows that the input representation doesn't reflect the underlying structure of language. We don't account for `film` and `movie` being closely related terms, in this case.

**Word Embeddings**: low-dimensional representation of words capturing their similarity. Low might be 50 - 300 dimensional. 

This might not seem low, but if the vocab size is 10,000 words, 50-300 is much lower, relatively.

If we plotted the vectors for the word embeddings, we'd want to see that similar words are grouped closely to one another.

This enables us to generalize to words that we may not have necessarily seen before. 

### How to learn embeddings

**Distributional Hypothesis** - Goes back to JR Firth (1957). 

> *"You shall know a word by the company it keeps"*.

Let's say we have access to a whole bunch of text on the web, how can we use that to learn what all these words mean?

```
I watched the movie
I watched the film
The film inspired me
The movie inspired me
```

The basic idea behind the distributional hypothesis is that film and movie could be substitutable in similar contexts. `Movie` and `film` are both direct objects of `watched`. `Watched` has *selectional preferences*, or certain types of arguments of things you can watch. Similar for subjects of the word `inspired`. 

```
I developed the film in the dark room
```

This is a context that is unique to film, because we have a different sense of the word (i.e. a film reel). There is complexity in terms of learning these things.

But, in aggregate, we expect to see `movie` and `film` in similar context more than something like `movie` and `mango`.

There have been applications, prior to NLP, like *Brown Clustering*.

### word2vec
Mokolov et al. 2013 created **word2vec**

Each word is mapped to a word vector and a context vector. We predict each word's context given that word. 

We map each word $w$ to a word vector $\overline{v}_w$ and a context vector $\overline{c}_w$.

We see a whole bunch of examples of words and context, then we learn vectors where a word's vector should be predictive of the words that are "around" places where that word is seen in the text.

Over time we'll learn a lot of ways of doing this, but over time we'll operationalize this idea of the distributional hypothesis and learn vectors that capture this kind of similarity.

## References
* [YouTube: A Complete Overview of Word Embeddings](https://www.youtube.com/watch?v=5MaWmXwxFNQ)
* 