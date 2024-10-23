from dataset.get_data import get_data_ml
from model.ml_model import SVM_train, predict, RandomForest_train, predict_
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help="output directory")
    parser.add_argument('-s', '--model_save_path', help="model save path")
    
    args = parser.parse_args()
    dir = args.dir
    model_save_path = args.model_save_path

    X_train_val, y_train_val, test_fold, X_test, y_test = get_data_ml(dir)

    #SVM
    # SVM_train(X_train_val, y_train_val,  test_fold, model_save_path)
    # predict(X_test, y_test, model_save_path)

    #Random Forest
    RandomForest_train(X_train_val, y_train_val,  test_fold, model_save_path)
    predict_(X_test, y_test, model_save_path)

