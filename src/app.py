import pandas as pd
from model import AirQualityClassifier
import inout
import time

SAVED_MODEL = '../model/classifier.sav'
TRAIN_DATA = '../data/train.csv'
TEST_DATA = '../data/test.csv'
PRED = '../predictions/predictions'

if __name__ == '__main__':
    classifier = AirQualityClassifier()
    classifier.load_data(TRAIN_DATA)
    print('Start training...\n')
    start = time.time()
    classifier.fit()
    print(f'... finished training in {time.time() - start} seconds.')
    print('Saving model...')
    inout.save_trained_model(SAVED_MODEL, classifier.model)
    print('Saved. Saving prediditions...')
    prediction = classifier.predict(TEST_DATA)
    inout.save_predictions(PRED, prediction)
    print('Saved. Done!')