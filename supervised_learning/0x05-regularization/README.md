## Regularization

The L1 and L2 regularization are widely used methods to control the model complexity and restrict over-fitting. There are some interesting comparisons between the L1 and L2 regularization. 


## Difference between L1 and L2 regularization

The main intuitive difference between the L1 and L2 regularization is that L1 regularization tries to estimate the median of the data while the L2 regularization tries to estimate the mean of the data to avoid overfitting.


The difference is that: while shrinking the quota, L1 tends to cut off some factors by turning their coefficients to zero, while L2 tend s to shrinking these coefficients to a tiny number (none zero), keep some of their influence on Y.


Another difference between them is that L1 regularization helps in feature selection by eliminating the features that are not important. This is helpful when the number of feature points are large in number.
