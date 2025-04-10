# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    # print("Training data:")
    # print("Shape:", train_df.shape)
    # print(train_df.head(2))
    # print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    # print("Test data:")
    # print(test_df.shape)
    # print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    # encode seasons
    # encoded_seasons_train = []
    # for i in range(len(train_df["season"])):
    #     encoded_seasons_train.append(encodeSeason(train_df["season"][i]))
    # encoded_seasons_test = []
    # for i in range(len(test_df["season"])):
    #     encoded_seasons_test.append(encodeSeason(test_df["season"][i]))

    # train_df = train_df.drop(["season"], axis=1)
    # train_df[["spring", "summer", "autumn", "winter"]] = pd.DataFrame(encoded_seasons_train)
    # test_df = test_df.drop(["season"], axis=1)
    # test_df[["spring", "summer", "autumn", "winter"]] = pd.DataFrame(encoded_seasons_test)

    # seasons = ["spring", "summer", "autumn", "winter"]
    # enc = OneHotEncoder()
    # train_df = enc.fit_transform(train_df)
    # print(train_df)

    # train_df["season"] = np.vectorize(lambda s: encodeSeason(s))(train_df["season"])
    # test_df["season"] = np.vectorize(lambda s: encodeSeason(s))(test_df["season"])

    onehot_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit on training data's season column
    onehot_enc.fit(train_df[['season']])

    # Transform both train and test
    season_train_encoded = onehot_enc.transform(train_df[['season']])
    season_test_encoded = onehot_enc.transform(test_df[['season']])

    # Create DataFrames with new column names
    season_train_df = pd.DataFrame(season_train_encoded, columns=onehot_enc.get_feature_names_out(['season']), index=train_df.index)
    season_test_df = pd.DataFrame(season_test_encoded, columns=onehot_enc.get_feature_names_out(['season']), index=test_df.index)

    # Drop old 'season' column and concatenate the encoded features
    train_df = pd.concat([train_df.drop(columns=['season']), season_train_df], axis=1)
    test_df = pd.concat([test_df.drop(columns=['season']), season_test_df], axis=1)

    print(train_df)

    # remove nan
    # for label in train_df.columns.values:
    #     train_df[label] = np.vectorize(lambda v: replaceNan(v, train_df[label]))(train_df[label])
    # for label in test_df.columns.values:
    #     test_df[label] = np.vectorize(lambda v: replaceNan(v, test_df[label]))(test_df[label])

    imp = IterativeImputer(tol=0.000001, max_iter=100, initial_strategy="mean", imputation_order="ascending")
    # combined_df = pd.concat([train_df.drop(['price_CHF'],axis=1), test_df])
    # imp = imp.fit(combined_df)

    # train_df_imp_without_target = imp.transform(train_df.drop(['price_CHF'],axis=1))
    # train_df_without_target = pd.DataFrame(train_df_imp_without_target, columns=train_df.drop(['price_CHF'],axis=1).columns)

    train_df = train_df.dropna(subset=["price_CHF"], axis=0)

    train_df_imp = imp.fit_transform(train_df)
    train_df = pd.DataFrame(train_df_imp, columns=train_df.columns)

    test_df_imp = imp.fit_transform(test_df)
    test_df = pd.DataFrame(test_df_imp, columns=test_df.columns)

    # train_df_without_target["price_CHF"] = train_df["price_CHF"]
    # train_df_imp = imp.fit_transform(train_df_without_target)
    # train_df = pd.DataFrame(train_df_imp, columns=train_df.columns)
    
    X_train = train_df.drop(['price_CHF'],axis=1).to_numpy()
    y_train = train_df['price_CHF'].to_numpy()
    X_test = test_df.to_numpy()

    # print(train_df)
    # print(test_df)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def encodeSeason(season):
    # match season:
    #     case "spring":
    #         return [1, 0, 0, 0]
    #     case "summer":
    #         return [0, 1, 0, 0]
    #     case "autumn":
    #         return [0, 0, 1, 0]
    #     case "winter":
    #         return [0, 0, 0, 1]
    # return [0, 0, 0, 0]
    match season:
        case "spring":
            return 0
        case "summer":
            return 1
        case "autumn":
            return 2
        case "winter":
            return 3
    return -1

def replaceNan(v, column):
    if (math.isnan(v)):
        return column.mean()
    return v

class Model(object):
    def __init__(self):
        super().__init__()

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        #TODO: Define the model and fit it using (X_train, y_train)
        avg = 0
        kf = KFold(n_splits=10)
        for i, (train, test) in enumerate(kf.split(X_train)):
            self.model = GaussianProcessRegressor(kernel=RationalQuadratic(), n_restarts_optimizer=3, alpha=0.1)
            self.model.fit(X_train[train], y_train[train])
            err = mean_squared_error(self.predict(X_train[test]), y_train[test]) ** 0.5
            print("fold", i, err)
            avg += err
        avg /= 10
        print("avg", avg)
        self.model = GaussianProcessRegressor(kernel=RationalQuadratic())
        self.model.fit(X_train, y_train)
        err = mean_squared_error(self.predict(X_train), y_train) ** 0.5
        print("all", err)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred=np.zeros(X_test.shape[0])
        #TODO: Use the model to make predictions y_pred using test data X_test
        y_pred = self.model.predict(X_test)

        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    model = Model()
    # Use this function for training the model
    model.train(X_train=X_train, y_train=y_train)

    # Use this function for inferece
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

