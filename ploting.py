from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from numpy.linalg import svd
import numpy as np

np.random.seed(0)
# np.random.seed(1)


def ITQ(X: np.array, n_bits, n_iterations):
    """Iterative Quantitization"""

    # Construct Orthogonal rotation matrix
    R = np.random.randn(n_bits, n_bits)
    [U, _, _] = svd(R)
    R = U[:, :n_bits]

    # Find Optimal Rotation
    for _ in range(n_iterations):
        V = X @ R
        [U, _, VT] = svd(np.sign(V).T @ X)
        R = (VT @ U.T)

    return R


X, y = make_circles(n_samples=3000, factor=0.4, noise=.2)

majority = np.where(y == 0)[0]
minority = np.where(y == 1)[0][:len(X) // 4]
X = np.concatenate((X[majority], X[minority]))
y = np.concatenate((y[majority], y[minority]))

labels, counts = np.unique(y, return_counts=True)

majIndexes = y == labels[np.argmax(counts)]
Xmaj = X[majIndexes]
ymaj = y[majIndexes]

R = ITQ(Xmaj, 2, 50)

# V x R
U = np.sign(Xmaj @ R).astype(np.int)

# Assign each sample to Hash Code Subspace
Q = np.packbits(np.where(U < 0, 0, U), axis=1, bitorder='little')

COLORS = {
    0: 'r',
    1: 'b',
    2: 'y',
    3: 'g'
}

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Space')
reds = y == 0
blues = y == 1

plt.scatter(X[reds, 0], X[reds, 1], c='red')
plt.scatter(X[blues, 0], X[blues, 1], c='blue')

plt.subplot(1, 3, 2)
plt.title('Majority Samples')
plt.scatter(Xmaj[:, 0], Xmaj[:, 1], c='red')

plt.subplot(1, 3, 3)
plt.title('Subspace Division')
for subspace in np.unique(Q):
    indexes = (Q == subspace).flatten()
    plt.scatter(Xmaj[indexes, 0], Xmaj[indexes, 1], c=COLORS[subspace])

plt.show()


