"""
Prepare fraud dataset for classification with xtreme boosted
classifier.

We're using a fraud dataset from:
https://www.kaggle.com/ntnu-testimon/paysim1/home

This code is heavily influenced by:
https://www.kaggle.com/arjunjoshua/predicting-fraud-in-financial-payment-services/
Thanks!
"""
import pandas as pd

def get_data():
    """
    Return a list of transaction features/variables (X) and
    an indicator if a transaction is fraud or not (Y).

    Our challange here is to get only relevant features.
    """
    data = pd.read_csv('data/transactions.csv')

    # We only need those two transaction types
    X = data.loc[(data.type == 'TRANSFER') | (data.type == 'CASH_OUT')]

    # Separate our fraud indicator (it is already encoded as 0
    # for (0 - legal transaction, 1 - fraud transaction)
    Y = X['isFraud']
    # Get rid of fraud indicator from our features.
    del X['isFraud']

    # Remove columns that currently don't add any
    # meaningful informations.
    X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)

    # Numericly encode the transaction type.
    X['type'] = X['type'].replace({'TRANSFER': 0, 'CASH_OUT': 1})

    # Remove all of the accounts where there's no money.
    # Again not adding any value (at lest at first sight.)
    # Removing the following give us boost from 70% accuracy to 87%
    X.drop(X.loc[(X.oldbalanceDest == 0) & (X.newbalanceDest == 0) & (X.amount != 0)], axis=1)
    X.drop(X.loc[(X.oldbalanceOrg == 0) & (X.newbalanceOrig == 0) & (X.amount != 0)], axis=1)

    return data, X, Y

if __name__ == '__main__':
    d, x, y=get_data()
