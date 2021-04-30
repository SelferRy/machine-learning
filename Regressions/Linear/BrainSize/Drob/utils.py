import numpy as np
import matplotlib.pyplot as plt


def init_param(m, n):
    # parameters = {}
    # L = len(layers_dims) - 1
    # n, m = X.shape[1], 1
    return np.random.randn(n, m)  # layers_dims[l], layers_dims[l - 1]) * np.sqrt(2. / layers_dims[l - 1])
    # for l in range(1, L + 1):
    #     parameters[f"W{l}"] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2. / layers_dims[l - 1])
    #     parameters[f"b{l}"] = np.zeros((layers_dims[l], 1))
    # return parameters


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
    print(1 / (2 * m) * np.sum(np.square(y_hat - y)))
    # y = y.reshape((-1, 1))
    #     print(f"compute_cost\n m = {m}, y_hat = {y_hat.shape}, y = {y.shape}")
    return 1 / (2 * m) * np.sum(np.square(y_hat - y))


# def init_adam(parameters):
#     L = len(parameters) // 2
#     v = {}
#     s = {}
#
#     # Initialize v, s. Input: "parameters". Outputs: "v, s".
#     for layer in range(L):
#         l = layer + 1
#         for dWb, Wb in zip(["dW", "db"], ["W", "b"]):
#             v[f"{dWb}{l}"] = np.zeros(parameters[f"{Wb}{l}"].shape)
#             s[f"{dWb}{l}"] = np.zeros(parameters[f"{Wb}{l}"].shape)
#     return v, s


def gradient_descent(parameters, grads, learning_rate=0.01):
    parameters -= learning_rate * grads
    # for Wb, dWb in zip("W", grads):
    #     print("Wb, dWb", Wb, dWb)
    #     parameters[Wb] -= learning_rate * grads[dWb]
    #     parameters["W1"] -= learning_rate * grads["dW1"]
    #     parameters["b1"] -= learning_rate * grads["db1"]
    return parameters


# def adam_optimizer(parameters, grads, v, s, t, learning_rate=0.01,
#                    beta1=0.9, beta2=0.999, epsilon=1e-8, mod="MSE"):
#     L = len(parameters) // 2
#     v_corrected = {}
#     s_corrected = {}
#
#     # if mod == "binary_cross_entropy":
#     #     for layer in range(L):
#     #         l = layer + 1
#     #         # p - params: W or b
#     #         for dWb, Wb in zip(["dW", "db"], ["W", "b"]):
#     #             v[f"{dWb}{l}"] = beta1 * v[f"{dWb}{l}"] + grads[f"{dWb}{l}"]
#     #             v_corrected[f"{dWb}{l}"] = v[f"{dWb}{l}"] / (1 - np.power(beta1, t))
#     #             s[f"{dWb}{l}"] = beta2 * s[f"{dWb}{l}"] + (1 - beta2) * np.square(grads[f"{dWb}{l}"])
#     #             s_corrected[f"{dWb}{l}"] = s[f"{dWb}{l}"] / (1 - np.power(beta2, t))
#     #             parameters[f"{Wb}{l}"] -= learning_rate * v_corrected[f"{dWb}{l}"]
#     #             parameters[f"{Wb}{l}"] /= np.sqrt(s_corrected[f"{dWb}{l}"]) + epsilon
#
#     if mod == "MSE" and L == 1:
#         for dWb, Wb in zip(["dW1", "db1"], ["W1", "b1"]):
#             #             print("adam_opt\nshapes: v, grads", v[dWb].shape, grads[dWb].shape)
#             v[dWb] = beta1 * v[dWb] + grads[dWb]
#             #             print("adam_optimizer\n", f"v[{dWb}] = {v[dWb]}", f"grads[{dWb}] = {grads[dWb]}")
#             v_corrected[dWb] = v[dWb] / (1 - np.power(beta1, t))
#             #             print("adam_optimizer\nv_corrected", v_corrected[dWb])
#             s[dWb] = beta2 * s[dWb] + (1 - beta2) * np.square(grads[dWb])
#             #             print("adam_optimizer\ns_corrected", s[dWb])
#             s_corrected[dWb] = s[dWb] / (1 - np.power(beta2, t))
#             parameters[Wb] -= learning_rate * v_corrected[dWb] / (np.sqrt(s_corrected[dWb]) + epsilon)
#     #             print("adam_optimizer\ns_corrected", s_corrected[dWb])
#     #             parameters[Wb] /= np.sqrt(s_corrected[dWb]) + epsilon
#
#     return parameters, v, s


def forward_prop(X, theta):
    """
    Forward propagation small neural network for Linear Regressions.
    Args:
        X: feature-vector.
        parameters: dict with W and b.

    Returns:
    y_hat (z_ or a_): output layer <=> our hypothesis.
    """
    # W = parameters["W1"]
    # b = parameters["b1"]
    # print("X, W, b shapes:\n", X.shape, W.shape, b.shape)
    return X.dot(theta)
    # cache = (y_hat, W, b)
    # return predict  # , cache


def backward_prop(X, y, y_hat):
    """ In the case it is just gradient. """
    m = X.shape[1]
    # predict = y_hat
    grad = 1/m * X.T.dot((y_hat - y))
    #     print("backprop\nshapes X, W, b, y:\n", X.shape, W.shape, b.shape, y.shape)
    # dW = 1. / m * X.T.dot(X.dot(W.T + b) - y)  # 2
    # db = 1. / m * np.sum((X.dot(W.T + b) - y), keepdims=True)
    #     print("backprop\nshapes dW, db:\n", dW.shape, db.shape)
    # gradients = {"dW1": dW.T, "db1": db}
    return grad # gradients


def model(X, Y, learning_rate=0.1, optimizer="gradient_descent", beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epochs=1000, print_cost=True):
    # L = len(layers_dims)
    costs = []
    t = 0
    m = X.shape[1]
    # Initialize parameters
    theta = np.random.randn(m, 1)

    # parameters = init_param_he(layers_dims)
    # print(parameters)
    #     W1, b1 = parameters["W1"], parameters["b1"]
    # v, s = init_adam(parameters)
    #     v, s = (adam_params["dW1"], adam_params["db1"]), (adam_params["dW1"], adam_params["db1"])
    #     print("model\nshapes of W1, b1, v, s:\n", W1.shape, b1.shape, v["dW1"].shape, v["db1"], s["dW1"], s["db1"])
    cost = 0  # cost_total
    # debug_cache = [parameters]

    # Optimization loop
    for i in range(num_epochs):
        # print(f"epochs = {i}\n")  # , parameters)
        y_hat = forward_prop(X, theta)  # , caches
        # print(f"cost_total = {cost}")
        # cost_total += compute_cost(z, Y)
        cost = compute_cost(y_hat, Y)
        grads = backward_prop(X, Y, y_hat)
        #         print("model\n", f"grads[{i}] = {grads}")
        if optimizer == "gradient_descent":
            theta = gradient_descent(parameters=theta, grads=grads, learning_rate=learning_rate)
        # if optimizer == "adam":
        #     t += 1
        #     theta, v, s = adam_optimizer(theta, grads, v, s, t,
        #                                       learning_rate, beta1, beta2, epsilon)

        # cost_avg = cost_total / m

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print(f"Cost after epoch {i}: {cost}")  # cost_avg
        if print_cost and i % 100 == 0:
            costs.append(cost)  # cost_avg

        # debug_cache.append((grads, parameters))

    plt.plot(costs)
    plt.ylabel('J')
    plt.xlabel('epochs (per 100)')
    plt.title(f"learning rate = {learning_rate}")
    plt.show()

    return theta, costs  # , debug_cache
