import pandas as pd
from model import AirQualityClassifier
from helpers import InOut
    
if __name__ == '__main__':
    classifier = AirQualityClassifier('../data/train.csv')
    in_out = InOut()
    classifier.fit()
    in_out.save_trained_model(classifier.model)
    prediction = classifier.predict('../data/test.csv')
