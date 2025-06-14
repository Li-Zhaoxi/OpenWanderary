import numpy as np
from scipy.special import softmax

val = np.array([1, 2, 3, 4, 5])
print(softmax(val))

# print(np.exp(val) / np.sum(np.exp(val)))
