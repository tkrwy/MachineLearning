from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, average_precision_score, roc_auc_score

def metric(y_test, y_pred, y_pred_prob):
    #accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy:{:.3f}".format(accuracy))
    #confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    print("confusion:", confusion)
    summary(confusion)
    #classification report
    report = classification_report(y_test, y_pred, target_names=["0", "1"])
    print("report:", report)
    #aupr
    aupr = average_precision_score(y_test, y_pred_prob)
    print("aupr:{: .3f}".format(aupr))
    #auroc
    auroc = roc_auc_score(y_test, y_pred_prob)
    print("auroc:{: .3f}".format(auroc))

def summary(confusion_matrix):
    TP = confusion_matrix[1, 1]
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
    Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
    Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
    F1_score = round(2 * Precision * Recall / (Recall + Precision),3) if Recall + Precision != 0 else 0.
    print("Precision:{: .3f}".format(Precision))
    print("Recall/sensitivity:{: .3f}".format(Recall))
    print("Specificity:{: .3f}".format(Specificity))
    print("F1_score:{: .3f}".format(F1_score))