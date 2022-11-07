# AIR QUALITY CLASSIFIER

> This program trains a RandomForest Classifier to get the air quality
> based on measures from a new type of sensor. 

## Background

Climate change is a major problem in the modern world. New technologies are 
being developed every day to fight against its effects. One of these new technologies
is a laser based sensor that measures the air quality. This ML model takes 
eight different values from the sensors and uses them to predicts the air quality.

## Results and analysis

Exponer y comentar los resultados obtenidos > 50 palabras (mín)

## Solution

The algorithm to use is a Random Forest. This algorithm is robust against overfitting and outliers, 
and works pretty well with non-linear data. It's easy to get a reasonable high accuracy score
without having to fine tune hyperparameters, specially if we use a Grid search algorithm, which
performs a cross validation over the dataset. This is what we've done in this project.

We first split the train dataset in train and test, to have some data that would allow us to evaluate
the model and the parameters passed to it. We calculated the f1 score, and visualized other metrics like
the confussion matrix (results are in `graphics.ipynb`). Seeing that the outcome is satisfying, we then 
retrained the model with the whole dataset. This was done based in the assumption that the more data
the model sees, the better trained will be. After that, we used the best estimator of the grid search
to make the final predictions (saved in the `/predictions` directory).

## Installation

Just run `pip install -r requirements`, and you're ready to go!

To train a new model, go in `/src` in the terminal and run `python app.py`. Or you can 
load the pretrained model saved in `/model`, using the `inout` module, and make new predictions.

## Contact info

Sebastián Dolgonos

[Linkedin](https://www.linkedin.com/in/sebastián-dolgonos-565733226/)

[Portfolio](https://bass2015.github.io)

[Github](https://github.com/Bass2015)

## License 

[MIT](https://opensource.org/licenses/MIT)


https://onlinelibrary.wiley.com/doi/full/10.1002/sam.11583