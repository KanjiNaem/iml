# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression

def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned. 

    Parameters
    ----------
    X: matrix of floats, dim = (135,13), inputs with 13 features
    y: array of floats, dim = (135,), input labels
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """
    weights = np.zeros((13,))
    
    model = Ridge(alpha = lam, fit_intercept = False).fit(X, y)
    weights = model.coef_

    assert weights.shape == (13,)
    return weights


def calculate_RMSE(w, X, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of 
    predicting y from X using a linear model with weights w. 

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression 
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    rmse: float: dim = 1, RMSE value
    """
    rmse = 0
    y_pred = np.dot(X, w)
    rmse = mean_squared_error(y, y_pred) ** 0.5
    
    assert np.isscalar(rmse)
    return rmse


def average_LR_RMSE(X, y, lambdas, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated, 
    and then averaged over iterations.
    
    Parameters
    ---------- 
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV
    
    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """
    RMSE_mat = np.zeros((n_folds, len(lambdas)))
    
    k_fold = KFold(n_splits = n_folds)
    for i, (train, test) in enumerate(k_fold.split(X)):
        for j, lam in enumerate(lambdas):
            weights = fit(X[train], y[train], lam)
            RMSE = calculate_RMSE(weights, X[test], y[test])
            RMSE_mat[i, j] = RMSE

    avg_RMSE = np.mean(RMSE_mat, axis = 0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")


