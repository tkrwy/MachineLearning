from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from dataset.get_data import get_data, get_data_header
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import pandas as pd
import numpy as np
import math
import argparse

from joblib import Parallel, delayed

# def random_forest(X_train, y_train):
#     max_features = (len(X_train.columns) - 1)*0.1
#     print(max_features = max_features)
#     select = SelectFromModel(RandomForestClassifier(n_estimators = 100, random_state=42), threshold=-np.inf, max_features=max_features)
#     select.fit(X_train, y_train)
#     X_train_l1 = select.transform(X_train)

#     print("X_train.shape: {}".format(X_train.shape))
#     print("X_train_l1.shape:{}".format(X_train_l1.shape))
#     mask = select.get_support()
#     plt.matshow(mask.reshape(1, -1), cmap='gray_r')
#     plt.xlabel("Sample index")
#     plt.show()
#     plt.savefig("fig/tnc_random_forest.png")
def select_from_model(model_name, X_train, y_train, fig_dir):
    max_features = math.floor(len(X_train.columns)*0.3)
    print("max_features", max_features)
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators = 100, random_state=42)
    elif model_name == "GradientBoostingClass":
        model =  GradientBoostingClassifier(random_state=8)
    elif model_name == "SVC":
        model = SVC(kernel="linear", random_state=2)
    elif model_name == "LogisticRegression":
        model = LogisticRegression()
    select = SelectFromModel(model, threshold=-np.inf, max_features=max_features)
    select.fit(X_train, y_train)
    X_train_l1 = select.transform(X_train)

    print("X_train.shape: {}".format(X_train.shape))
    print("X_train_l1.shape:{}".format(X_train_l1.shape))
    mask = select.get_support()
    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel("feature index")
    plt.show()
    plt.savefig(fig_dir + "_" + model_name + "_feature_visual.png")
    return mask

def feature_selection(X_train, y_train, fig_dir):
    
    features = X_train.columns.values
    
    model_name = "LogisticRegression"
    mask = select_from_model(model_name, X_train, y_train, fig_dir)
    print("LogisticRegression selected features:",features[mask])
    set1 = set(features[mask].tolist())

    model_name = "RandomForest"
    mask = select_from_model(model_name, X_train, y_train, fig_dir)
    print("RandomForest selected features:",features[mask])
    set2 = set(features[mask].tolist())

    model_name = "GradientBoostingClass"
    mask = select_from_model(model_name, X_train, y_train, fig_dir)
    print("GradientBoostingClass selected features:",features[mask])
    set3 = set(features[mask].tolist())

    intersection = set1&set2&set3
    print("intersection:", intersection)
    print("intersection len", len(intersection))
    selected_columns = list(intersection)

   
    plt.figure(figsize=(16,10))
    venn3([set1, set2, set3], set_labels=["LogisticRegression", "RandomForest",  "GradientBoostingClass"])
    plt.savefig(fig_dir+ "_venn.png")
    print(features[mask])

    return selected_columns

def feature_selection_for_task(in_dir, suffix, out_dir, fig_dir):
    train_file = in_dir + "train" + "_" + suffix +".csv"
    X_train, y_train = get_data_header(train_file, 0)
    selected_columns = feature_selection(X_train, y_train, fig_dir)
    
    X_train_selected = pd.DataFrame(X_train, columns=selected_columns)
    X_train_selected["labels"] = y_train
    X_train_selected.to_csv(out_dir + "train.csv", index=False)

    splits = ["val", "test"]
    for split in splits:
        file = in_dir + split + "_"+ suffix + ".csv"
        X, y = get_data(file)
        X_selected = pd.DataFrame(X, columns=selected_columns)
        X_selected["labels"] = y
        X_selected.to_csv(out_dir + split + ".csv", index=False)

def parallel_processing(tasks):
    num_cores = 7
    Parallel(n_jobs=num_cores)(delayed(feature_selection_for_task)(task) for task in tasks)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--in_dir', help="input directory")
    parser.add_argument('-s', '--suffix', help="input file suffix, eg. for train_preprocessed.csv, the suffix is _preprocessed ")
    parser.add_argument('-o', '--out_dir', help="output directory")
    parser.add_argument('-f', '--fig', help="figure output directory")
    
    args = parser.parse_args()
    in_dir = args.in_dir
    suffix = args.suffix
    out_dir = args.out_dir
    fig_dir = args.fig

    print("dir", dir)

    feature_selection_for_task(in_dir, suffix, out_dir, fig_dir)

    print("feature selection done!")

    
