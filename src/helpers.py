import pandas as pd
import numpy as np
import pickle
import json
from sklearn_evaluation import plot

class InOut():
    def __init__(self):
        pass

    def save_trained_model(self, path, model):
        """Saves a trained model as pickle file."""
        with open(path, 'wb') as file:
            pickle.dump(model, file)
    
    def load_model_from_pickle(self, path):
        """Loads a pretrained model"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def save_predictions(self, filename, predictions):
        if isinstance(predictions, np.ndarray):
            self.__save_as_csv(f'{filename}.csv', predictions)
            self.__save_as_json(f'{filename}.json', predictions)
        else:
            raise TypeError('predictions must be of class numpy.ndarray')

    def __save_as_csv(self, path, predictions):
        np.savetxt(path, predictions, delimiter=';', fmt='%i', header='target')
    
    def __save_as_json(self, path, predictions):
        pred_dict = {'target':{}}
        pred_dict['target'] = dict(enumerate(predictions))
        with open(path, 'w') as file:
            json.dump(pred_dict, file, cls=NpEncoder)  
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

