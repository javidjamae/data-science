# PAC Learning and Sample Complexity

The Probably Approximately Correct (PAC) learning framework is a fundamental concept in computational learning theory that describes what it means for a learning algorithm to perform well. It forms a bridge to [Mistake-Bounded Learning (MBL)](./mistake-bounded-learning.md) by providing a mathematical foundation for defining and bounding mistakes within learning algorithms. Below are the core aspects of the PAC framework, including its relation to sample complexity and achievable mistake bounds.

## Definition of Learnability
PAC learning defines a concept or hypothesis as being learnable if there exists an algorithm that can find a close approximation to it with high probability, given enough data.

## Error Bounds
In PAC learning, the learner aims to find a hypothesis that is approximately correct with respect to the true underlying distribution. This means that the error of the hypothesis (the probability that it disagrees with the true concept) is bounded by a small positive value, often denoted by $\epsilon$.

## Confidence Level
PAC learning involves a confidence parameter, often denoted by $\delta$, defining the probability that the learning algorithm's output hypothesis might not meet the required error bound. The algorithm must find a hypothesis with error less than $\epsilon$ with probability at least $1 - \delta$.

## Polynomial Time
PAC learning requires that the learning process is efficient, meaning that the algorithm must run in polynomial time concerning the size of the input, the complexity of the hypothesis class, $ 1/\epsilon$, and $1/\delta$.

## Distribution-Free Learning
PAC learning is considered distribution-free, meaning that the guarantees hold regardless of the underlying distribution of the data, assuming it is drawn independently and identically from that distribution.