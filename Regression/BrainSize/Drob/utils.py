import numpy as np

def init_param_he(layers_dims):
    parameters = {}
    L = len(layers_dims) - 1
    for l in range(1, L + 1):
        parameters[f"W{l}"] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2./layers_dims[l - 1])
        parameters[f"b{l}"] = np.zeros((layers_dims[l], 1))
    return parameters


def compute_cost(X, y, parameters):
    W = parameters["W1"]
    b = paraneters["b1"]

    y_hat = np.dot(X, W) + b
    J = 1/(2*m) * np.sum(np.square(y_hat - y))
    return J


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


def adam(parameters, grads, v, s, t, learning_rate = 0.01,
         beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for layer in range(L):
        l = layer + 1
        # p - params: W or b
        for dWb, Wb in zip(["dW", "db"], ["W", "b"]):
            v[f"{dWb}{l}"] = beta1 * v[f"{dWb}{l}"] + grads[f"{dWb}{l}"]
            v_corrected[f"{dWb}{l}"] = v[f"{dWb}{l}"] / (1 - np.power(beta1, t))
            s[f"{dWb}{l}"] = beta2 * s[f"{dWb}{l}"] + (1 - beta2) * np.square(grads[f"{dWb}{l}"])
            s_corrected[f"{dWb}{l}"] = s[f"{dWb}{l}"] / (1 - np.power(beta2, t))
            parameters[f"{Wb}{l}"] -= learning_rate * v_corrected[f"{dWb}{l}"]
            parameters[f"{Wb}{l}"] /= np.sqrt(s_corrected[f"{dWb}{l}"]) + epsilon

    return parameters, v, s
