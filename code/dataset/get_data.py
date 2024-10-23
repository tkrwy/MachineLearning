import pandas as pd
import numpy as np
def get_data(file):
    df = pd.read_csv(file, header = 0)
    features = df.iloc[:,1:-1]
    
    
    # df.loc[df['label']=="human_eRNAbase_eRNAs"] = 1
    # df.loc[df['label']=="human_eRNAbase_eRNAs"] = 0
    label = df.iloc[:, -1]
    return features, label

def get_sampleName(file):
    df = pd.read_csv(file, header = 0)
    
    eRNAName = df.iloc[:, 0]
    return eRNAName

def load_data(file, header = 0):
    read_chunks = pd.read_csv(file, header=header, iterator=True,chunksize=65535)
    chunk_list = list()
    for chunk in read_chunks:
        chunk_list.append(chunk)
        # print(chunk, type(chunk))
    df = pd.concat(chunk_list, axis=0, ignore_index=False)
    return df

def get_data_header(file, header = 0):
    # df = pd.read_csv(file, header = header)
    read_chunks = pd.read_csv(file, header=header, iterator=True,chunksize=65535)
    chunk_list = list()
    for chunk in read_chunks:
        chunk_list.append(chunk)
        # print(chunk, type(chunk))
    df = pd.concat(chunk_list, axis=0, ignore_index=False)
    
    features = df.iloc[:,1:-1]
    
    
    # df.loc[df['label']=="human_eRNAbase_eRNAs"] = 1
    # df.loc[df['label']=="human_eRNAbase_eRNAs"] = 0
    label = df.iloc[:, -1]
    return features, label
def get_data_ml(file_dir):
    train_file = file_dir + "train.csv"
    X_train, y_train = get_data(train_file)

    val_file = file_dir + "val.csv"
    X_val, y_val = get_data(val_file)

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])    

    test_fold = np.zeros(len(X_train_val))
    test_fold[:len(X_train)] = -1
    
    test_file = file_dir + "test.csv"
    X_test, y_test = get_data(test_file)

    return X_train_val, y_train_val, test_fold, X_test, y_test

def get_data_from_db(base_dir):
    file_dir = base_dir +"/"
    train_file = file_dir + "train.csv"
    X_train, y_train = get_data(train_file)

    val_file = file_dir + "val.csv"
    X_val, y_val = get_data(val_file)

    test_file = file_dir + "test.csv"
    X_test, y_test = get_data(test_file)
    return  X_train, y_train, X_val, y_val, X_test, y_test 

def get_sampleName_from_db(base_dir):
    file_dir = base_dir +"/"
    train_file = file_dir + "train.csv"
    eRNAName_train = get_sampleName(train_file)

    val_file = file_dir + "val.csv"
    eRNAName_val = get_sampleName(val_file)

    test_file = file_dir + "test.csv"
    eRNAName_test = get_sampleName(test_file)
    return  eRNAName_train, eRNAName_val, eRNAName_test

if __name__ == "__main__":
    task = "tnc"
    file_dir = "/home/storage/lldata/human/SEQ/eRNAbase/"+ task +"/"
    train_file = file_dir + "train.csv"
    X_train, y_train = get_data(train_file)

    val_file = file_dir + "val.csv"
    X_val, y_val = get_data(val_file)

    X_train_val = pd.concat([X_train, X_val])
    y_tain_val = pd.concat([X_train, y_train])

    test_file = file_dir + "test.csv"
    X_test, y_test = get_data(test_file)

    