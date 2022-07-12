"""
Plot result
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve, recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
def show_result(threshold,result,label):
    result = np.squeeze(result, axis=1)
    precision_data, recall_data, _ = sklearn.metrics.precision_recall_curve(label, result)
    AP_value = sklearn.metrics.average_precision_score(label, result)
    
    #plot Precision-Recall Graph
    plt.figure()
    plt.title("Precision-Recall Graph")
    plt.xlabel("Recall"   )
    plt.ylabel("Precision")
    plt.plot(recall_data, precision_data, "b", label = "AP = %0.4F" % AP_value)
    plt.legend(loc = "upper right")
    plt.show()
    #plot ROC curve
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, result)
    AUC_value=sklearn.metrics.auc(fpr, tpr)
    pred=result>threshold
    cm=confusion_matrix(label, pred)

    plt.figure()
    plt.title("ROC Curve")
    plt.plot(fpr, tpr,'r', label = 'AUC = %0.2f' %AUC_value)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.legend(loc = 'lower right')
    plt.show()
    #print metrics
    print("confusion matrix:\n",cm)
    print("accuracy_score: {}".format( accuracy_score(label, pred)))
    print("precision_score: {}".format( precision_score(label, pred)))
    print("recall_score: {}".format( recall_score(label, pred)))
    print("f1_score: {}".format(f1_score(label, pred)))
    print("AUC:",sklearn.metrics.auc(fpr, tpr))