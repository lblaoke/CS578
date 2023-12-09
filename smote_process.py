from imblearn.over_sampling import SMOTE
import pandas as pd

def read_data(datafilename, hasIndex=True):
    if hasIndex:
        return pd.read_csv(datafilename, index_col=0)
    else:
        return pd.read_csv(datafilename)

def write_data(datafilename, X, Y, hasIndex=True):
    data = pd.concat([X, Y], axis=1)
    data.to_csv(datafilename, index=hasIndex)

def process_imbalance(X, Y):
    X_balance, Y_balance = SMOTE().fit_resample(X, Y)
    
    X_balance_df = pd.DataFrame(X_balance, columns=X.columns)
    Y_balance_df = pd.DataFrame(Y_balance, columns=['TARGET']) 
    write_data('balanced_train.csv', X_balance_df, Y_balance_df)
    return X_balance_df, Y_balance

df = read_data("processed_data.csv")
df_test = df.sample(frac=0.1)
df_train = df.drop(df_test.index)

df_test.to_csv('balanced_test.csv', index=False)

cols = df_train.columns
Y = df_train['TARGET'] # label
X = df_train[cols.drop('TARGET')] # features
X_balance, Y_balance = process_imbalance(X,Y)
