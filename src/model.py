import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score

SAVED_MODEL = '../model/classifier.sav'

class AirQualityClassifier():
    def load_data(self, train_file_path):
        """Reads the train csv and splits the data in train and test"""
        self.train_df = pd.read_csv(train_file_path, sep=';')
        self.splitted_data = self.split_data()

    def split_data(self):
        """Splits the training data in train and validation datasets.
        Split ratio is decided according to """
        split_ratio = 1/((len(self.train_df.columns)-1)**0.5 +1)
        data = self.train_df.drop('target', axis=1)
        target = self.train_df['target']
        return train_test_split(data, target, test_size=split_ratio)

    def fit(self):
        """Performs a grid search to find the best model."""
        params = {'max_depth': range(10, 18, 2), 
                        'n_estimators': 1000,
                        'criterion':['gini', 'entropy'], 
                        'max_features': ['sqrt', 0.33,0.4,0.66,0.8,1]
        }
        X_train, _, y_train, _ = self.splitted_data
        self.model = GridSearchCV(RandomForestClassifier(),
                            params, 
                            cv=5,
                            verbose=True)
        self.model.fit(X_train, y_train)
  
    def predict(self, data_path, sep=';'):
        """Reads an unlabeled csv file and makes a prediction."""
        data = pd.read_csv(data_path, sep=sep)
        return self.model.best_estimator_.predict(data)
    
    def train_score(self):
        _, X_test, _, y_test = self.splitted_data
        y_pred = self.predict(X_test)
        return f1_score(y_test, y_pred, average='macro') 