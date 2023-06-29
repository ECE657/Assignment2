import numpy as np

def relu(X):
    """
    ReLU function activation
    Params:
        X as matrix
    Return:
        X if X > 0 else 0
    """
    return(np.maximum(0, X))

def sigmoid(X):
    """
    Sigmoid function activation
    Params:
        X as matrix
    Return:
        1 / (1 + np.exp(-X))
    """
    return 1 / (1 + np.exp(-X))

def tanh(X):
    """
    Hyperbolic tangent function activation
    Params:
        X as matrix
    Return:
        (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
    """
    return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

def softmax(X):
    """
    Softmax function for a matrix or vector X
    Params:
        X: numpy array
    Returns:
        softmax output as a numpy array
    """
    softmax_output = np.exp(X) / np.exp(X).sum(axis=1, keepdims=True)      
    return softmax_output

def gaussian(dj, sigma):
    return np.exp(-(dj/(2*sigma))**2)

# Test cases
if __name__ == "__main__":
    X = np.array([-1, 2, 0, -3, 4])

    # Test case for ReLU
    expected_relu = np.array([0, 2, 0, 0, 4])
    result_relu = relu(X)
    assert np.array_equal(result_relu, expected_relu), "ReLU test case failed"

    # Test case for Sigmoid
    expected_sigmoid = np.array([0.26894142, 0.88079708, 0.5, 0.04742587, 0.98201379])
    result_sigmoid = sigmoid(X)
    assert np.allclose(result_sigmoid, expected_sigmoid), "Sigmoid test case failed"

    # Test case for Tanh
    expected_tanh = np.array([-0.76159416, 0.96402758, 0., -0.99505475, 0.9993293])
    result_tanh = tanh(X)
    assert np.allclose(result_tanh, expected_tanh), "Tanh test case failed"

    # Test case for Softmax
    expected_softmax = np.array([5.80206892e-03, 1.16537670e-01, 1.57716585e-02, 7.85224641e-04, 8.61103378e-01])
    result_softmax = softmax(X)
    print(result_softmax)
    assert np.allclose(result_softmax, expected_softmax), "Softmax test case failed"
    
    print("All test cases passed!")