import matplotlib.pyplot as plt

from sklearn import metrics

def confusion_matrix(actual,predicted,label1,label2):

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[label1, label2])

    cm_display.plot()
    plt.show()