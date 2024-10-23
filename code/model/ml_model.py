import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from pandarallel import pandarallel
import joblib
from .metrics import metric

def SVM(X_train_val, y_train_val,  test_fold, X_test, y_test, model_save_path):    

    if not os.path.exists(model_save_path):
        model = SVC(random_state=2, probability=True)
        ps = PredefinedSplit(test_fold = test_fold)
        params_search = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
        grid_search_params = {
            'estimator': model,
            'param_grid': params_search,
            'cv': ps,
            'n_jobs': 36,
            'verbose': 32
        }
        grsearch = GridSearchCV(**grid_search_params)
        grsearch.fit(X_train_val, y_train_val)
        joblib.dump(grsearch, model_save_path)
    else:
        grsearch = joblib.load(model_save_path)
    bst = grsearch.best_estimator_
    print('bst:', bst)
    test_score = grsearch.score(X_test, y_test)
    print('test accuracy: {: .3f}'.format(test_score))


    y_pred = grsearch.predict(X_test)
    # y_pred_prob = grsearch.predict_proba(X_test)[:, 1]
    y_pred_prob = grsearch.decision_function(X_test)

    metric(y_test, y_pred, y_pred_prob)

def SVM_train(X_train_val, y_train_val,  test_fold, model_save_path):    

    if not os.path.exists(model_save_path):
        model = SVC(random_state=2, probability=True)
        ps = PredefinedSplit(test_fold = test_fold)
        params_search = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
        grid_search_params = {
            'estimator': model,
            'param_grid': params_search,
            'cv': ps,
            'n_jobs': 36,
            'verbose': 32
        }
        grsearch = GridSearchCV(**grid_search_params)
        grsearch.fit(X_train_val, y_train_val)
        joblib.dump(grsearch, model_save_path)
    else:
        grsearch = joblib.load(model_save_path)

def predict(X_test, y_test, model_save_path):    

    grsearch = joblib.load(model_save_path)
    bst = grsearch.best_estimator_
    print('bst:', bst)
    test_score = grsearch.score(X_test, y_test)
    print('test accuracy: {: .3f}'.format(test_score))

    y_pred = grsearch.predict(X_test)
    # y_pred_prob = grsearch.predict_proba(X_test)[:, 1]
    y_pred_prob = grsearch.decision_function(X_test)

    metric(y_test, y_pred, y_pred_prob)

def RandomForest(X_train_val, y_train_val,  test_fold, X_test, y_test, model_save_path):
    if not os.path.exists(model_save_path):
        model = RandomForestClassifier(random_state=2)
        ps = PredefinedSplit(test_fold = test_fold)
        params_search = {'n_estimators': [5, 10, 15]}
        grid_search_params = {
            'estimator': model,
            'param_grid': params_search,
            'cv': ps,
            'n_jobs': 3,
            'verbose': 32
        }
        grsearch = GridSearchCV(**grid_search_params)
        grsearch.fit(X_train_val, y_train_val)

        joblib.dump(grsearch, model_save_path)
    else:
        grsearch = joblib.load(model_save_path)
    bst = grsearch.best_estimator_
    print('bst:', bst)
    test_score = grsearch.score(X_test, y_test)
    print('test accuracy: {: .3f}'.format(test_score))

    y_pred = grsearch.predict(X_test)
    y_pred_prob = grsearch.predict_proba(X_test)[:, 1]

    metric(y_test, y_pred, y_pred_prob)

def RandomForest_train(X_train_val, y_train_val,  test_fold, model_save_path):
    if not os.path.exists(model_save_path):
        model = RandomForestClassifier(random_state=2)
        ps = PredefinedSplit(test_fold = test_fold)
        params_search = {'n_estimators': [5, 10, 15]}
        grid_search_params = {
            'estimator': model,
            'param_grid': params_search,
            'cv': ps,
            'n_jobs': 3,
            'verbose': 32
        }
        grsearch = GridSearchCV(**grid_search_params)
        grsearch.fit(X_train_val, y_train_val)

        joblib.dump(grsearch, model_save_path)
    else:
        grsearch = joblib.load(model_save_path)

def predict_(X_test, y_test, model_save_path):
    grsearch = joblib.load(model_save_path)
    bst = grsearch.best_estimator_
    print('bst:', bst)
    test_score = grsearch.score(X_test, y_test)
    print('test accuracy: {: .3f}'.format(test_score))

    y_pred = grsearch.predict(X_test)
    y_pred_prob = grsearch.predict_proba(X_test)[:, 1]

    metric(y_test, y_pred, y_pred_prob)