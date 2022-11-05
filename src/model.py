import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score

TRAIN_DATA = '../data/train.csv'

class AirQualityClassifier():
    """
    Class that takes a labeled dataset and trains on it, using 
    a RandomForest algorithm.
    
    The class actually uses a Grid Search to look for the best 
    combination of hyperparameters for the Random Forest. It's possible 
    also to load a pretrained grid as a model, specifying it in the 
    constructor.
    
    Attributes
    --------
    train_df: pd.DataFrame
        A DataFrame containing labeled data for training.
    splitted_data: list
        A list containing 4 dataframes: X_train, X_test, y_train, y_test
    model: sklearn.model_selection.GridSearchCV
        The trained model. If no model was provided using the constructor, 
        this attributes will be generated after using the fit() method
    """
    def __init__(self, model: GridSearchCV=None) -> None:
        """
        Parameters
        ---------
        model: GridSearchCV
            A trained model that can be used to make predictions. 
            Default -> None. In this case, it will be necessary to 
            train a new model.
        Raises
        -----
        TypeError
            If the model passed as parameter is not of class GridSearchCV
        """
        if (model is None or 
            isinstance(model, GridSearchCV)):
            self.model = model
            self.load_data(TRAIN_DATA)
            return
        raise TypeError('model must be of class sklearn.model_selection.GridSearchCV')

    def load_data(self, train_file_path: str):
        """
        Reads a train csv and splits the data in train and test
        
        Parameters
        ----------
        train_file_path: str
            The csv file path to load
        """
        self.train_df = pd.read_csv(train_file_path, sep=';')
        self.splitted_data = self.split_data()

    def split_data(self) -> list:
        """
        Splits the training data in train and validation datasets.
        
        Returns
        -------
        list
            A list containing 4 dataframes: X_train, X_test, y_train, y_test
        """
        split_ratio = 1/((len(self.train_df.columns)-1)**0.5 +1)
        data = self.train_df.drop('target', axis=1)
        target = self.train_df['target']
        return train_test_split(data, target, test_size=split_ratio)

    def fit(self):
        """
        Performs a grid search to find the best model and saves the model as
        a class attribute
        """
        params = {'max_depth': range(10, 18, 2), 
                        'n_estimators': [1000],
                        'criterion':['gini', 'entropy'], 
                        'max_features': ['sqrt', 0.33,0.4,0.66,0.8,1]
        }
        X_train, _, y_train, _ = self.splitted_data
        self.model = GridSearchCV(RandomForestClassifier(),
                            params, 
                            cv=5,
                            verbose=True)
        self.model.fit(X_train, y_train)
        self.f1score = self.train_score
  
    def predict(self, filepath: str, sep: str=';') -> np.ndarray:
        """
        Reads an unlabeled csv file and makes a prediction.
        
        Parameters
        ---------
        filepath: str
            The path where the csv file is located.
        sep: str
            The separator used in the csv file. Default: ';'

        Returns
        ---------
        np.ndarray
            A 1D NumPy array containing the predicted labels
        """
        if self.model is None:
            raise ValueError("""Model not trained, or object initialized with no model.
                                Please fit the classifier to make a prediction.""")
        data = pd.read_csv(filepath, sep=sep)
        return self.model.best_estimator_.predict(data)
    
    def train_score(self) -> float:
        """
        F1 score(macro) of the best model, based on test split of 
        the dataset used to train the model. 
        
        Returns
        -------
        float
            The train f1 score
        """
        _, X_test, _, y_test = self.splitted_data
        y_pred = self.predict(X_test)
        return f1_score(y_test, y_pred, average='macro') 