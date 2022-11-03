import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score

SAVED_MODEL = './model/classifier.sav'

class AirQualityClassifier():
    def load_data(self, train_file_path):
        """Reads the train csv and splits the data in train and test"""
        self.train_df = pd.read_csv('./data/train.csv', sep=';')
        self.splitted_data = self.split_data()

    def split_data(self):
        """Splits the training data in train and validation datasets.
        Split ratio is decided according to """
        split_ratio = 1/((len(self.train_df.columns)-1)**0.5 +1)
        data = self.train_df.drop('target', axis=1)
        target = self.train_df['target']
        return train_test_split(data, target, test_size=split_ratio)

    def fit(self):
        """Performs a grid search cross validation to find the 
        best model."""
        params = {'max_depth': range(10, 18, 2), 
                        'n_estimators': 1000,
                        'criterion':['gini', 'entropy'], 
                        'max_features': ['sqrt', 0.33,0.4,0.66,0.8,1]
        }
        X_train, _, y_train, _ = self.splitted_data
        grid = GridSearchCV(RandomForestClassifier(),
                            params, 
                            cv=5,
                            verbose=True)
        grid.fit(X_train, y_train)
        self.estimator = grid.best_estimator_
    
    def save_model(self):
        """Saves the best model as pickle file."""
        with open(SAVED_MODEL, 'wb') as file:
            pickle.dump(self.estimator, file)
    
    def load_model(self):
        """Loads a pretrained model"""
        with open(SAVED_MODEL, 'rb') as f:
            self.estimator = pickle.load(f)
  
    def predict(self, data):
        return self.estimator.predict(data)
    
    def train_score(self):
        _, X_test, _, y_test = self.splitted_data
        y_pred = self.predict(X_test)
        return f1_score(y_test, y_pred, average='macro') 