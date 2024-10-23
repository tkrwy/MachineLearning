import random
import pandas as pd
import math
import numpy as np
import argparse

def split_train_val_test_chunk(file,header):
    read_chunks = pd.read_csv(file, header=header, iterator=True,chunksize=65535)
    chunk_list = list()
    for chunk in read_chunks:
        chunk_list.append(chunk)
        # print(chunk, type(chunk))
    df = pd.concat(chunk_list, axis=0, ignore_index=False)
    
    row_count = df.shape[0]

    row_index_list = np.arange(row_count)
    random.seed(123)
    random.shuffle(row_index_list)

    senven_percent = math.floor(row_count*0.7)
    nine_percent = math.floor(row_count*0.9)

    train_set_row_index = np.arange(0,senven_percent,1)
    val_set_row_index = np.arange(senven_percent, nine_percent,1)
    test_set_row_index = np.arange(nine_percent, row_count)

    train_set = df.iloc[row_index_list[train_set_row_index]]
    val_set = df.iloc[row_index_list[val_set_row_index]]
    test_set = df.iloc[row_index_list[test_set_row_index]]
    
    return train_set, val_set, test_set
def split_train_val_test(file):
    df = pd.read_csv(file, header=0)

    row_count = df.shape[0]

    row_index_list = np.arange(row_count)
    random.seed(123)
    random.shuffle(row_index_list)

    senven_percent = math.floor(row_count*0.7)
    nine_percent = math.floor(row_count*0.9)

    train_set_row_index = np.arange(0,senven_percent,1)
    val_set_row_index = np.arange(senven_percent, nine_percent,1)
    test_set_row_index = np.arange(nine_percent, row_count)

    train_set = df.iloc[row_index_list[train_set_row_index]]
    val_set = df.iloc[row_index_list[val_set_row_index]]
    test_set = df.iloc[row_index_list[test_set_row_index]]
    
    return train_set, val_set, test_set

def split(input_file, dir):
    
    train_set, val_set, test_set = split_train_val_test_chunk(input_file, 0)

    train_set.to_csv(dir+"train.csv", index=False)
    val_set.to_csv(dir+"val.csv", index=False)
    test_set.to_csv(dir+"test.csv", index=False)

#input: csv
#output: train.csv val.csv test.csv

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input file")
    parser.add_argument('-d', '--dir', help="output directory")
    
    args = parser.parse_args()
    input = args.input
    dir = args.dir

    split(input, dir)

    print("data split done!")
    


    


