import pytest
# TODO: add necessary import
import numpy as np
from ml.model import inference
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics

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
def test_uses_rf():
    """
    this test ensures the RandomForest algorithm is used
    """
    #mock data
    X = np.array([[1,2], [3,4]])
    y = np.array([0,1])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


# TODO: implement the third test. Change the function name and input as needed
def test_divide_zero():
    """
    test if we handle worst case scenario of zero positives in compute_model_metrics function

    because we set zero_division=1 we should assert 1.0 = 1.0
    """
    #mock data
    true = np.array([1, 1, 1, 1])
    pred = np.array([0, 0, 0, 0])
    
    #compute_model_metrics
    precision, recall, fbeta = compute_model_metrics(true, pred)

    #should be one for precision
    assert precision == 1.0
    #should be zero for recall (opposite)
    assert recall == 0.0
    #should be 0 for fscore because recall is 0 and precision * 0 = 0
    assert fbeta == 0.0
