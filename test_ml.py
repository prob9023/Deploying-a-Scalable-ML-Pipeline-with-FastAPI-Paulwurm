import pytest
# TODO: add necessary import
import numpy as np
from ml.model import inference
from sklearn.ensemble import RandomForestClassifier

# TODO: implement the first test. Change the function name and input as needed
def test_inference_array():
    """
    test that the inference function returns an array rather than another type
    """
    #mock model
    model = RandomForestClassifier().fit(np.array([[1,2], [3,4]]), np.array([0,1]))
    X_test = np.array([[5,6]])

    #test inference
    preds = inference(model,X_test)

    #make assertion that it's an array
    assert isinstance(preds, np.ndarray)


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
