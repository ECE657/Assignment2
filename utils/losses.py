import numpy as np

def l1_loss(y_true, y_pred):
    """
    L1 loss (Mean Absolute Error)
    Params:
        y_true: numpy array, true values
        y_pred: numpy array, predicted values
    Returns:
        mean absolute error (MAE)
    """
    return np.mean(np.abs(y_true - y_pred))

def l2_loss(y_true, y_pred):
    """
    L2 loss (Mean Squared Error)
    Params:
        y_true: numpy array, true values
        y_pred: numpy array, predicted values
    Returns:
        mean squared error (MSE)
    """
    return np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    y_true = np.array([2, 2, 2, 2, 2])
    y_true = np.expand_dims(y_true, axis=0)
    y_pred = np.array([0, 0, 0, 0, 0])
    y_pred = np.expand_dims(y_pred, axis=0)

    # Test case for l1_loss
    expected_l1_loss = 2.0
    result_l1_loss = l1_loss(y_true, y_pred)
    assert np.isclose(result_l1_loss, expected_l1_loss), f"l1_loss test case failed, getting result_l1_loss = f{result_l1_loss}"

    # Test case for l2_loss
    expected_l2_loss = 4.0
    result_l2_loss = l2_loss(y_true, y_pred)
    assert np.isclose(result_l2_loss, expected_l2_loss), f"l2_loss test case failed, getting result_l2_loss = {result_l2_loss}"

    print("All test cases passed!")