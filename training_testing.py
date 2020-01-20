"""
Must install XGBClassfier to use this script:
conda install -c conda-forge xgboost
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score 
from xgboost.sklearn import XGBClassifier
from prep import get_data

def train_test(X, Y):
     """
     Train and test the data, show the accuracy of the model.
     """
     # Split dataset
     trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2)
     print("Train dataset contains {0} rows and {1} columns".format(trainX.shape[0], trainX.shape[1]))
     print("Test dataset contains {0} rows and {1} columns".format(testX.shape[0], testX.shape[1]))
    


     weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())
     xgb = XGBClassifier(scale_pos_weight = weights, n_jobs=4)
     xgb_fitted = xgb.fit(trainX, trainY)
     # Train the model on the test set to measure accuracy.
     proba = xgb_fitted.predict_proba(testX)
     preds = xgb_fitted.predict(testX)



     #REMEMBER THE DIFFERENCE BETWEEN PRECISION AND ACCURACY



     #creating metric to gauge model reliability
     #accuracy score compaires testY and predvalues
     #average_precision_score compaires testY and scoreY
     tree_accuracy = accuracy_score(testY, preds)
     tree_log_loss = log_loss(testY, proba)
     tree_average_precision=average_precision_score(testY, proba[:, 1])
     
     
     print("== XGB Decision Tree Metrics==")
     print("Accuracy: {0:.2f}".format(tree_accuracy))
     print("Log loss: {0:.2f}".format(tree_log_loss))
     print('Model accuracy (AUPRC) = {:.2f}%'.format(tree_average_precision*100))
     #print("Number of nodes created: {}".format(xgb.tree.node_count))


     #correct = 0

     #for i in range(len(preds)):
               #if (testY[i] == preds[i]):
                    #correct += 1
     #print("Predicted correctly: {0}/{1}".format(correct, len(preds)))
     print("Error: {0:.4f}".format(1-tree_accuracy))



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
               tree_average_precision))
     plt.show()


     #from xgboost import plot_tree
     #import graphviz
     # taking the model without .predict_proba(testX)
     #plot_tree(xgb)




if __name__ == '__main__':
     #grabs data from funtion in prep.py
     _, X, Y=get_data()
     train_test(X, Y)
     #from xgboost import plot_tree
     #import graphviz
     ##taking the model without .predict_proba(testX)
     #plot_tree(xgb)
