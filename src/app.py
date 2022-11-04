import pandas as pd
from model import AirQualityClassifier
import inout
    
if __name__ == '__main__':
    classifier = AirQualityClassifier()
    classifier.load_data('../data/train.csv')
    classifier.fit()
    inout.save_trained_model(classifier.model)
    prediction = classifier.predict('../data/test.csv')
    inout.save_trained_model()
