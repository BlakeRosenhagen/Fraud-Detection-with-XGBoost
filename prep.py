
import pandas as pd

def get_data():
   """
    Return a list of transaction features/variables (X) and
    an indicator if a transaction is fraud or not (Y).
    """
    data = pd.read_csv('data/transactions.csv')
    #Only transaction types that will be used are as follows.
    X = data.loc[(data.type == 'TRANSFER') | (data.type == 'CASH_OUT')]
    #distinguish fraud indicator
    Y = X['isFraud']
    #remove
    del X['isFraud']

    #columns that are not useful
    X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)

    #encode numerically
    X['type'] = X['type'].replace({'TRANSFER': 0, 'CASH_OUT': 1})
    #Remove accounts with no money, as its not needed, and improves accuracy
    X.drop(X.loc[(X.oldbalanceDest == 0) & (X.newbalanceDest == 0) & (X.amount != 0)], axis=1)
    X.drop(X.loc[(X.oldbalanceOrg == 0) & (X.newbalanceOrig == 0) & (X.amount != 0)], axis=1)

    return data, X, Y

if __name__ == '__main__':
    d, x, y=get_data()
    print("Data Preparation Complete")
