# Naive bayes classifier for movie review sentiment analysis

Prediction of a new review is done with
`python3 predict_movie_label.py write your own review here. it is pretty handy isn\'t it`

## Details
When `naivebayes.py` is run, preprocessing steps are selected with 
cross validation on the training set.
The best pipeline+model then gets trained on the full training set, 
and we calculate accuracy and F1 based on the testing set.
Then the best pipeline+model gets trained on the full training+test set 
and pickled in the models/ folder.

Using `predict_movie_label.py` then unpickles the models/ and predicts 
the label of the newly input reivew

## Preprocessing steps

## Author(s)
- Malte Lau Petersen, maltelau@protonmail.com, github.com/maltelau
