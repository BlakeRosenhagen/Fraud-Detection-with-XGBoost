# UT-Final-Project
Training a machine learning model to identify whether particular transactions are fraud.
## Attention!

#### Download csv from https://www.kaggle.com/ntnu-testimon/paysim1 and place csv file in "data" folder. Title file "transaction.csv"

Make sure XGBClassifier is installed: conda install -c conda-forge xgboost

Also, you may have to revert matplotlib library back to 1.5:

* pip uninstall matplotlib

* pip install matplotlib==1.3.1



## Introduction
The primary library that I have chosen to use is the XGBoost library, which I believe is somehow connected to Sklearn.
More specifically, the type of machine learning that is functioning is a combonation between logistical regression and gradiant boosting trees.
Logistical regression is optimal for binary factors such as in this case of whether a transaction can be classified as fraud or not.



## Usability
So after ensuring that xgboost is installed, open the jupyter notebook.

It will be a very simple file since I have been practicing "compartimentalizing" the code into seperate files that can be called
upon when needed. Just run the whole file or each cell individually. Fair warning, it may take upwards of five minuits to calculate everything involved.



## Results of the code
#### The main two parts of the output are as follows:

The XGB Decision Tree Metrics, which shows key statistics on the performance of the model, such as the Accuracy, Log Loss, and Error.

The graphical representation of the precision recall curve which the tradeoff between precision and recall are shown.
Precision is true positives over true positives plus false positives, while recall is true positives over true positives plus false negatives.

#### Example: https://github.com/BlakeRosenhagen/UT-Final-Project/blob/master/images/example.PNG

#### If for some reason there are no error messages and the graph does not show then create a cell above with this:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"



## Explanation of the code
The jupyter file starts out by running "prep". This is a file that takes in the "transactions.csv" and cleans it so it is ready for processing.

Next "training_testing" is ran which calls upon the "get_data" function in "prep" to collect the x and y needed

Spliting the data into four different groups called trainX, testX, trainY, and testY is then fitted into the xgbclassifier.

The probability predictions and regular predictions are generated and used to be the basis of three different metrics which I got from https://scikit-learn.org/stable/modules/classes.html

Then the "pickle" library is used to save the model into "models" which can be called upon later for prediction.

A really cool visulaization is created with the next section of code, which I found on https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py

To read more on this: https://stats.stackexchange.com/questions/350134/how-does-gradient-boosting-calculate-probability-estimates

There is some code in "training_testing.py" which I was experimeting with, but could not figure out how to function. Its in there so I can go back later and tweak.

## Technologies used
* Python
* Jupyter Notebook
* Pandas
* Numpy
* SciKit-Learn
* XGBoost
* Pickle
* Matplotlib
* Inspect
* Graphviz (not used)
