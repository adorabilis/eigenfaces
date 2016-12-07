import os
import numpy as np
from PIL import Image

def read_images(path):
    print "Reading " + path
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except:
                    print "Read error."
            c = c + 1
    print "Read successful (" + str(len(X)) + " images)."
    return [X, y]

def asRowMatrix(X):
    mat = np.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
    print "row matrix is "
    print mat.shape
    return mat

def pca(X, y, noOfComponents=0):
    print np.shape(X)
    print np.shape(y)
    [n, d] = X.shape
    if (noOfComponents <= 0) or (noOfComponents > n):
        noOfComponents = n
    mu = X.mean(axis=0)
    X = X - mu
    if n > d:
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in xrange(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[0:noOfComponents].copy()
    eigenvectors = eigenvectors[:, 0:noOfComponents].copy()
    print eigenvectors.shape
    print eigenvalues.shape
    print mu.shape
    return [eigenvalues, eigenvectors, mu]

def project(W, X, mu=None):
    if mu is None:
        return np.dot(X, W)
    return np.dot(X - mu, W)

def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y, W.T)
    return np.dot(Y, W.T) + mu

def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

class Eigenface(object):

    def __init__(self, X=None,y=None):
        self.noOfComponents = 0
        self.projections = []
        self.W = []
        self.mu = []
        self.compute(X, y)

    def compute(self, X, y):
        [D, self.W, self.mu] = pca(asRowMatrix(X), y, self.noOfComponents)
        self.y = y
        for xi in X:
            self.projections.append(project(self.W, xi.reshape(1, -1), self.mu))

    def predict(self, X):
        minDist = np.finfo('float').max
        minClass = -1
        Q = project(self.W, X.reshape(1, -1), self.mu)
        for i in xrange(len(self.projections)):
            p = np.asarray(self.projections[i]).flatten()
            q = np.asarray(Q).flatten()
            dist = np.sqrt(np.sum(np.power((p-q),2)))
            if dist < minDist:
                minDist = dist
                minClass = self.y[i]
        return minClass

[X, y] = read_images("/Users/dan/Downloads/at")
print np.shape(X)
print np.shape(y)
model = Eigenface(X[1:], y[1:])
print "expected =", y[100], "/", "predicted =", model.predict(X[200])



