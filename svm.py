import numpy as np
from sklearn.datasets import load_digits
from sklearn import svm
import matplotlib.pyplot as plt

digits = load_digits()
clf = svm.SVC()
X = digits.data[:-10]
y = digits.target[:-10]
clf.fit(X, y)


print("Prediction", clf.predict(digits.data[-1].reshape(1, -1)))
plt.imshow(digits.images[-1],  cmap=plt.get_cmap('gray'), interpolation="nearest")
plt.show()