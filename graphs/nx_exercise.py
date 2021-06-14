import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric
from sklearn.model_selection import train_test_split



path = "./data/"
filename = path + 'graph-000000139945-ffmpeg-clang-asan-O3_g.dot-ff_j_rev_dct2.pickle'


is_gcc = '-gcc-' in filename
is_clang = '-clang-' in filename
print('Compiled by', 'gcc' * is_gcc + 'clang' * is_clang)


g = nx.read_gpickle(filename)
d = torch_geometric.utils.from_networkx(g)


# class DumbModel:
#
#     def fit(...):
#         pass
#
#     def predict(data: np.ndarray):
#         return np.zeros((data.shape[0], ))

# train_ind,
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
