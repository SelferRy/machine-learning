import glob
import numpy as np


def get_feature_labels(path="./", extension="pickle"):
    """
    Get tuple with filenames (links to the feature vectors X) and labels (vector Y).
    Args:
        path: path to dir
        extension: sought extension of files.

    Returns:
    (filenames_array, labels-vector)
    """
    filenames = glob.glob(f'{path}*.{extension}')
    is_gcc = lambda filename: '-gcc-' in filename
    labels = [0 if is_gcc(filename) else 1 for filename in filenames]
    labels = np.array(labels).reshape(-1, 1)
    filenames = np.char.array(filenames).reshape(-1, 1)
    return filenames, labels  #  np.concatenate((filenames, labels), axis=1)


def splitter(filenames, labels):
    ind = np.arange(len(filenames))
    ind = np.random.permutation(ind)
    X_train, X_val, X_test = np.split(filenames[ind], [int(.6 * len(filenames)), int(.8 * len(filenames))])
    Y_train, Y_val, Y_test = np.split(labels[ind], [int(.6 * len(labels)), int(.8 * len(labels))])
    return (X_train, Y_train),\
           (X_val, Y_val), \
           (X_test, Y_test)


X, Y = get_feature_labels("./data/")
train, val, test = splitter(X, Y)

if __name__ == "__main__":
    for x, y in zip([train[0], val[0], test[0]], [train[1], val[1], test[1]]):
        print(x.shape, y.shape)
