## Regularization

The L1 and L2 regularization are widely used methods to control the model complexity and restrict over-fitting. There are some interesting comparisons between the L1 and L2 regularization. 


## Difference between L1 and L2 regularization

If both L1 and L2 regularization work well, you might be wondering why we need both. It turns out they have different but equally useful properties. From a practical standpoint, L1 tends to shrink coefficients to zero whereas L2 tends to shrink coefficients evenly. L1 is therefore useful for feature selection, as we can drop any variables associated with coefficients that go to zero. L2, on the other hand, is useful when you have collinear/codependent features. Codependence tends to increase coefficient variance, making coefficients unreliable/unstable, which hurts model generality. L2 reduces the variance of these estimates, which counteracts the effect of codependencies.


The main intuitive difference between the L1 and L2 regularization is that L1 regularization tries to estimate the median of the data while the L2 regularization tries to estimate the mean of the data to avoid overfitting.


The difference is that: while shrinking the quota, L1 tends to cut off some factors by turning their coefficients to zero, while L2 tend s to shrinking these coefficients to a tiny number (none zero), keep some of their influence on Y.


Another difference between them is that L1 regularization helps in feature selection by eliminating the features that are not important. This is helpful when the number of feature points are large in number.


**Interesting resource:**
- [3 The difference between L1 and L2 regularization](https://explained.ai/regularization/L1vsL2.html#:~:text=L1%20is%20therefore%20useful%20for,you%20have%20collinear%2Fcodependent%20features.).
- [L1 vs L2 Regularization: The intuitive difference](https://medium.com/analytics-vidhya/l1-vs-l2-regularization-which-is-better-d01068e6658c).
