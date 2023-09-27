# Sentiment Analysis

These are my notes from [this lecture](https://www.youtube.com/watch?v=cKbnEmjxnOY&list=PLofp2YXfp7TZZ5c7HEChs0_wfEfewLDs7&index=9) that is part of [this course](https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html). 

## Lecture Notes

Sentiment analysis is an instance of a binary classification problem. 

Example
```
this movie was great! would watch again [+]
```

Using bag-of-words features on unigrams, let's you capture words like `great` for positive sentiment. And bi-grams (pairs of words) allow us to capture pairs like `watch again`.

Other examples may be more difficult, however:
```
the movie was gross and overwrought, but I liked it [+]

this movie was not really very enjoyable [-]
```

In the first example, here `gross` and `overwrought`, but then says `but` which overrides everything before it.

And `enjoyable` is negated in the second example. 

Bag-of-words doesn't seem sufficient (discourse structure, negation).

There are some ways around this, for example when using bi-grams, we could extract bigram feature for `not X` for all `X` following the `not`.

