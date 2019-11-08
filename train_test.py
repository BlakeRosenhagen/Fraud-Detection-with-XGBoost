"""
Train and test XGBClassifier on financial fraud data.

We have a few chalanges here:
1) Much less fraud transactions than the legal ones (inbalanced dataset)
   - we need to choose an algorithm that will easily handle it - XGBClassfier
     and configure it correctly
2) Quite a lot of data - around 500MB
   - it would take a lot of time to do a cross validation method
     in a similar way that we did in Section3 so we need to validate
     our model on a test set.

Note:
You need to innstall XGBClassfier to use this script:
conda install -c conda-forge xgboost
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from prep import get_data

def train_test(X, Y):
     """
     Train and test the data, show the accuracy of the model.
     """
     # Split dataset into 80% training set and 20% test set.
     trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2)
     # Create a model and configure it correctly for
      # imbalanced dataset by setting up scale_pos_weight
      # (ratio of number of negative class to the positive class)
     weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())
     xgb = XGBClassifier(scale_pos_weight = weights, n_jobs=4)
     # Train the model on the test set to measure accuracy.
     proba = xgb.fit(trainX, trainY).predict_proba(testX)
     average_precision=average_precision_score(testY, proba[:, 1])
     print('Model accuracy (AUPRC) = {:.2f}%'.format(average_precision*100))


     import pickle
     pickle.dump(xgb, open("models/pima.pickle.dat", "wb"))

     #load model later on
     #xgb_model_loaded = pickle.load(open(file_name, "rb"))





     from sklearn.metrics import precision_recall_curve
     import matplotlib.pyplot as plt
     from inspect import signature

     precision, recall, _ = precision_recall_curve(testY, proba[:, 1])

     # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
     step_kwargs = ({'step': 'post'}
                    if 'step' in signature(plt.fill_between).parameters
                    else {})
     plt.step(recall, precision, color='b', alpha=0.2,
          where='post')
     plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

     plt.xlabel('Recall')
     plt.ylabel('Precision')
     plt.ylim([0.0, 1.05])
     plt.xlim([0.0, 1.0])
     plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
               average_precision))
     plt.show()


     #from xgboost import plot_tree
     #import graphviz
     ## taking the model without .predict_proba(testX)
     #model = xgb.fit(trainX, trainY)
     #plot_tree(model)



if __name__ == '__main__':
     _, X, Y=get_data()
     train_test(X, Y)
