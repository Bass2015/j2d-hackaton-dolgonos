import inout
import time
from model import AirQualityClassifier

SAVED_MODEL = '../model/classifier.sav'
TEST_DATA = '../data/test.csv'
PRED = '../predictions/predictions'

if __name__ == '__main__':
    classifier = AirQualityClassifier()

    print('Start training...\n')
    start = time.time()
    classifier.fit()
    print(f'... finished training in {time.time() - start} seconds.')
    print(f"""The model has a f1 score of {classifier.train_score()} 
            with the train data""")
    classifier.calculate_confusion_matrix()
    print('Saving model...')
    inout.save_trained_model(SAVED_MODEL, classifier)
    
    print('Saved. Saving prediditions...')
    prediction = classifier.predict_from_csv(TEST_DATA)
    inout.save_predictions(PRED, prediction)
    print('Saved. Done!')