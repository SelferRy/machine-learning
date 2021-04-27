import numpy as np
import matplotlib.pyplot as plt

def init_param_he(layers_dims):
    parameters = {}
    L = len(layers_dims) - 1
    for l in range(1, L + 1):
        parameters[f"W{l}"] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2./layers_dims[l - 1])
        parameters[f"b{l}"] = np.zeros((layers_dims[l], 1))
    return parameters


def compute_cost(y_hat, y):
    """
    Compute Mean Square Error (MSE).
    Args:
        y_hat: our hypothesis after forward_prop
        y: labels-vector

    Returns:
    J: cost value of MSE loss function.
    """
    m = y_hat.shape[0]
    # y = y.reshape((-1, 1))
    # print(y_hat.shape, y.shape)
    return 1/(2*m) * np.sum(np.square(y_hat - y))


def init_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for layer in range(L):
        l = layer + 1
        for dWb, Wb in zip(["dW", "db"], ["W", "b"]):
            v[f"{dWb}{l}"] = np.zeros(parameters[f"{Wb}{l}"].shape)
            s[f"{dWb}{l}"] = np.zeros(parameters[f"{Wb}{l}"].shape)
    return v, s


def adam_optimizer(parameters, grads, v, s, t, learning_rate = 0.01,
         beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, mod="MSE"):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    # if mod == "binary_cross_entropy":
    #     for layer in range(L):
    #         l = layer + 1
    #         # p - params: W or b
    #         for dWb, Wb in zip(["dW", "db"], ["W", "b"]):
    #             v[f"{dWb}{l}"] = beta1 * v[f"{dWb}{l}"] + grads[f"{dWb}{l}"]
    #             v_corrected[f"{dWb}{l}"] = v[f"{dWb}{l}"] / (1 - np.power(beta1, t))
    #             s[f"{dWb}{l}"] = beta2 * s[f"{dWb}{l}"] + (1 - beta2) * np.square(grads[f"{dWb}{l}"])
    #             s_corrected[f"{dWb}{l}"] = s[f"{dWb}{l}"] / (1 - np.power(beta2, t))
    #             parameters[f"{Wb}{l}"] -= learning_rate * v_corrected[f"{dWb}{l}"]
    #             parameters[f"{Wb}{l}"] /= np.sqrt(s_corrected[f"{dWb}{l}"]) + epsilon

    if mod == "MSE" and L == 1:
        for dWb, Wb in zip(["dW1", "db1"], ["W1", "b1"]):
            print("adam_opt\nshapes: v, grads", v[dWb].shape, grads[dWb].shape)
            v[dWb] = beta1 * v[dWb] + grads[dWb]
            v_corrected[dWb] = v[dWb] / (1 - np.power(beta1, t))
            s[dWb] = beta2 * s[dWb] + (1 - beta2) * np.square(grads[dWb])
            s_corrected[dWb] = s[dWb] / (1 - np.power(beta2, t))
            parameters[Wb] -= learning_rate * v_corrected[dWb]
            parameters[Wb] /= np.sqrt(s_corrected[dWb]) + epsilon

    return parameters, v, s


def forward_prop(X, parameters):
    """
    Forward propagation small neural network for Linear Regression. 
    Args:
        X: feature-vector.
        parameters: dict with W and b.

    Returns:
    y_hat (z_ or a_): output layer <=> our hypothesis.
    """
    W = parameters["W1"]
    b = parameters["b1"]
    # print("X, W, b shapes:\n", X.shape, W.shape, b.shape)
    z = np.dot(X, W.T) + b
    cache = (z, W, b)
    return z, cache


def backward_prop(X, y, cache):
    """ In the case it is just gradient. """
    m = X.shape[1]
    z, W, b = cache
    print("backprop\nshapes X, W, b, y:\n", X.shape, W.shape, b.shape, y.shape)
    dW = 2./m * X.T.dot(X.dot(W.T + b) - y)
    db = 2./m * np.sum((X.dot(W.T + b) - y), keepdims=True)
    print("backprop\nshapes dW, db:\n", dW.shape, db.shape)
    gradients = {"dW1": dW, "db1": db}
    return gradients


def model(X, Y, layers_dims, learning_rate=0.01, beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epochs=10000, print_cost=True):
    # L = len(layers_dims)
    costs = []
    t = 0
    m = X.shape[1]

    # Initialize parameters
    parameters = init_param_he(layers_dims)
    v, s = init_adam(parameters)
    cost_total = 0

    # Optimization loop
    for i in range(num_epochs):
        z, caches = forward_prop(X, parameters)
        cost_total += compute_cost(z, Y)
        grads = backward_prop(X, Y, caches)
        t += 1
        parameters, v, s = adam_optimizer(parameters, grads, v, s, t,
                                          learning_rate, beta1, beta2, epsilon)

    cost_avg = cost_total / m

    # Print the cost every 1000 epoch
    if print_cost and i % 1000 == 0:
        print(f"Cost after epoch {i}: {cost_avg}")
    if print_cost and i % 100 == 0:
        costs.append(cost_avg)

    plt.plot(costs)
    plt.ylabel('J')
    plt.xlabel('epochs (per 100)')
    plt.title(f"learning rate = {learning_rate}")
    plt.show()

    return parameters
