import inout
import time
from model import AirQualityClassifier

TEST_MODEL = '../model/test_classifier.sav'
FINAL_MODEL = '../model/final_classifier.sav'
TEST_DATA = '../data/test.csv'
PRED = '../predictions/predictions'
TRAIN_DATA = '../data/train.csv'

def train_classifier(classifier, filepath, test=False):
    """
    Trains an AirQualityClassifier.
    
    Parameters
    ---------
    classifier: AirQualityClassifier
        The classifier to train
    filepath: str
        The filepath where the model will be saved as pickle
    test: bool
        If set to True, the train dataset will be splited in train
        and test. Use it to evaluate the model
    """
    classifier.load_data(TRAIN_DATA)
    
    print('Start training...\n')
    start = time.time()
    classifier.fit()
    print(f'... finished training in {time.time() - start} seconds.')
    if test:
        score = classifier.train_score()
        classifier.calculate_confusion_matrix()
    else:
        score = classifier.model.best_estimator_.score
    print(f"""The model has a score of {score} 
            with the train data""")
    print('Saving test model...')
    inout.save_trained_model(filepath, classifier)
    print('Saved')

def make_predictions(final_classifier):
    print('Saved. Saving predictions...')
    prediction = final_classifier.predict_from_csv(TEST_DATA)
    inout.save_predictions(PRED, prediction)
    print('Saved. Done!')

if __name__ == '__main__':
    print('Training on splited dataset...')
    test_classifier = AirQualityClassifier()
    train_classifier(test_classifier, TEST_MODEL, test=True)

    print('Training on the whole dataset...')
    final_classifier = AirQualityClassifier()
    train_classifier(final_classifier, FINAL_MODEL)
    
    make_predictions(TEST_DATA, PRED, final_classifier)
