
import matplotlib.pyplot as plt
import numpy as np
tsne = __import__('8-tsne').tsne

np.random.seed(0)
X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
Y = tsne(X, perplexity=50.0, iterations=3000, lr=750)
plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.colorbar()
plt.title('t-SNE')
plt.show()

"For comparison, here’s how PCA performs on the same dataset:"

pca = __import__('1-pca').pca

X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
Y = pca(X, 2)
plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.colorbar()
plt.title('PCA')
plt.show()