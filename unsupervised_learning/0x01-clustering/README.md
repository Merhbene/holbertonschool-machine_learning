# **Clustering**

##  K-Means:
K-Means is a popular non-probabilistic clustering algorithm. The goal of the algorithm is to minimize the distortion measure J. We achieve that by the following iterative procedure:
* Choose the number of clusters K
* Initialize the vector μ_k that defines a central point of each cluster
* Assign each data point x to the closest cluster centre
* Recalculate central points μ_k for each cluster
* Repeat 3–4 until central points stop moving


![1_ik3r8uZgzGVGA-bgQVIyaw](https://miro.medium.com/max/640/1*lPcP9mUtfq9sApyWtPIvQg.gif)


## Gaussian Mixtures:
Gaussian Mixtures are based on K independent Gaussian distributions that are used to model K separate clusters.
The most important thing to know about GMs is that the convergence of this model is based on the EM (expectation-maximization) algorithm. It is somewhat similar to K-Means and it can be summarized as follows:
* Initialize μ, ∑, and mixing coefficient π and evaluate the initial value of the log likelihood L
* Evaluate the responsibility function using current parameters
* Obtain new μ, ∑, and π using newly obtained responsibilities
* Compute the log likelihood L again. Repeat steps 2–3 until the convergence.
The Gaussian Mixtures will also converge to a local minimum.

![Convergence of Gaussian Mixtures.](https://miro.medium.com/max/623/1*kJYirC6ewCqX1M6UiXmLHQ.gif)

## K-Means vs GMM:
The first visible difference between K-Means and Gaussian Mixtures is the shape the decision boundaries. GMs are somewhat more flexible and with a covariance matrix ∑ we can make the boundaries elliptical, as opposed to circular boundaries with K-means.
Another thing is that GMs is a probabilistic algorithm. By assigning the probabilities to datapoints, we can express how strong is our belief that a given datapoint belongs to a specific cluster.

If we compare both algorithms, the Gaussian mixtures seem to be more robust. However, GMs usually tend to be slower than K-Means because it takes more iterations of the EM algorithm to reach the convergence. They can also quickly converge to a local minimum that is not a very optimal solution.
