import pandas as pd
from sklearn.impute import KNNImputer
from dataset.get_data import get_data_from_db, get_sampleName_from_db
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import argparse

def preprocess(base_dir):
    X_train, y_train, X_val, y_val, X_test, y_test = get_data_from_db(base_dir)
    eRNAName_train, eRNAName_val, eRNAName_test = get_sampleName_from_db(base_dir)

    X_columns = X_train.columns.tolist()
    print("X_columns len:", len(X_columns))
   

    #drop columns with nan
    # cols_with_nan = df.columns[df.isna().any().tolist()]
    # print("na columns", len(cols_with_nan), cols_with_nan)
    # df1 = df.dropna(axis=1, how='any')
    # df1.to_csv("/home/lulu/eRNAHunter/lldata/human/SEQ/eRNAbase/eRNAbase.hg38.mathfeatures.seq.na.csv", index=False)

    # #impute missiing values
    # imputer = KNNImputer(n_neighbors = 3, weights="uniform")
    # X_train_impute = imputer.fit_transform(X_train)

    # #scaler
    # scaler = MinMaxScaler()
    # X_train_scaler = scaler.fit_transform(X_train_impute)

    #pipeline
    pipe = Pipeline([(
        'imputer', KNNImputer(n_neighbors = 3, weights="uniform")),
        ('scaler', MinMaxScaler())])
    X_train_proc = pipe.fit_transform(X_train) 

    X_val_proc = pipe.transform(X_val)
    X_test_proc = pipe.transform(X_test)

    eRNAName_train.name = "name"
    train = pd.DataFrame(X_train_proc, columns=X_columns)
    y_train.name = "label"
    df_train = pd.concat([eRNAName_train, train, y_train], axis=1)
    df_train.to_csv(base_dir + "/train_preprocessed.csv", index=False)

    eRNAName_val.name = "name"
    val = pd.DataFrame(X_val_proc, columns=X_columns)
    y_val.name = "label"
    df_val = pd.concat([eRNAName_val, val, y_val], axis=1)
    df_val.to_csv(base_dir +  "/val_preprocessed.csv", index=False)

    eRNAName_test.name = "name"
    test = pd.DataFrame(X_test_proc, columns=X_columns)
    y_test.name = "label"
    df_test = pd.concat([eRNAName_test, test, y_test], axis=1)
    df_test.to_csv(base_dir + "/test_preprocessed.csv", index=False)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help="output directory")
    
    args = parser.parse_args()
    dir = args.dir
    print("dir", dir)

    preprocess(dir)
   

    


