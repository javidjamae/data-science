# Skip-gram

These are my notes from [this lecture](https://www.youtube.com/watch?v=hznxqCIrzSQ&list=PLofp2YXfp7TZZ5c7HEChs0_wfEfewLDs7&index=21) that is part of [this course](https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html).

## Lecture Notes

The skip-gram model is a way to create word embeddings.

**Input:** The input for this model is a large corpus of sentences. We can scrape a large amount of data from the web to do this. We may have some issues pulling in sensitive data, or data we don't want our model to have, but this is generally the approach.

**Output:** The output is $\overline{v}_w, \overline{c}_w$ for each word type $w$.

**Hyperparams:** 
* $d$ - word vector dimension (~50 or ~300)
* $k$ - window size


### How this works
Assume $k=1$

Example sentence:
```
The film inspired
```

We're going to create a mapping by taking all neighbors of each word token up to $k$ positions away.

| word    | context  |
|---------|----------|
| film    | inspired |
| film    | The      |

Skip-gram is a probabilistic model of context given a word.

$$
P( \text{context}=y | \text{word}=x ) = \frac{exp(\overline{v}_x \cdot \overline{c}_y)}{\sum\limits_{y^i \epsilon V} exp(\overline{v}_x \cdot \overline{c}_{y^i})}
$$

Where $\overline{v}, \overline{c}$ are model parameters (each is a weight matrix) of size $|V| \times d$.

Where:
* $V$ is the number of rows of vectors for the words
* $d$ is the number of dimensions for each word embedding

The number of parameters in our model is:

$$
2 \cdot |V| \times d
$$

So, if $\overline{v}_x$ is similar to $\overline{c}_y$ then $y$ is likely to be in $x$'s context.

If $\overline{v}_x$ and $\overline{c}_y$ are closely aligned then their dot product will be large. When we exponentiate the dot product we're going to get a high score and we'll assign it a high probability. So, if $\overline{v}_x$ for one word is similar to $\overline{c}_y$ for another word is indicative of those words occuring near each other. 

### Example

Corpus:
```
I saw
```

Parameters:
* $d=2$
* $\overline{v}$
* $\overline{c}$

In practice we randomly assign all these parameters and run gradient descent. But, for the example, let's just assume we have a starting $\overline{v}$ that looks like this:


$$
\overline{v}_{\text{I}} = [1, 0]
$$
$$
\overline{v}_{\text{saw}} = [0, 1]
$$

We have two word / context pairs:

| word    | context |
|---------|---------|
| I       | saw     |
| saw     | I       |


If $\overline{c}_{\text{saw}} = [1,0]$ and $\overline{c}_{\text{I}} = [0,1]$, what is $P(\text{context}|\text{word}=\text{saw})$?

This means that the vectors for $\overline{c}_{\text{saw}}$ and $\overline{v}_{\text{I}}$ are the same, and the vectors for $\overline{v}_{\text{saw}}$ and $\overline{c}_{\text{I}}$ are the same.

Then we can calculate:

$$
exp(\overline{v}_{\text{saw}} \cdot \overline{c}_{\text{saw}}) = 1
$$

We intuitively know this because they are orthoganal vectors on the a 2d graph.

And,

$$
exp(\overline{v}_{\text{saw}} \cdot \overline{c}_{\text{I}}) \approx 3
$$

So,
$$
P(\text{context}=\text{I}|\text{word}=\text{saw})=\frac{3}{4}
$$

$$
P(\text{context}=\text{saw}|\text{word}=\text{saw})=\frac{1}{4}
$$

This confirms our intuition where we should have "I" more likely to happen given "saw" than the other way around. 

The other thing it shows you is that the word space and context space are not the same. In this case, they're almost rotations of each other. The words should be close to the context of the words they're related, but not necessarily to the words themselves. For example, $\overline{v}_{\text{saw}}$ and $\overline{c}_{\text{I}}$ should be close together, but $\overline{v}_{\text{saw}}$ and $\overline{v}_{\text{I}}$ should not. 

### Training

Based on our window size k, we extract word and context pairs. In training, we maximize the sum over these pairs of the log probability of the context given the word:

$$
\sum\limits_{(x,y)} \log P(\text{context}|\text{word}=x)
$$

So, we:
* Scrape the web
* With $k=1$ extract adjacent words
* Form a training set of word/context pairs
* maximize the sum of the log probability of the observed pairs

This is an "impossible" problem. Can't drive the probability to 1. We're never going to be able completely optimize this objective from the standpoint of getting every prediction to be probability 1, because there are going to be many words that appear in the context of a given word x. So you'll have conflicting training examples.

Unlike classifications, where you can (given a big enough neural network) fit the data perfectly, here that's not going to happen. 

We're going to initialize our parameters randomly. 

## References
* [Distributed Representations of Words and Phrases
and their Compositionality](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)