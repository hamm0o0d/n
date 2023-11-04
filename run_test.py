
import numpy as np
from Models.perceptron import Perceptron
from Models.adaline import Adaline
from HelperFunctions.train_test_split import train_test_split
from HelperFunctions.load_dataset import load_dataset


def runTest(chosenModel, chosen_features, class1, class2, learningRate, epochsNum, mseThreshold, addBias):

    df = load_dataset()

    ppp = df[(df['Class'] == class1) | (df['Class'] == class2)]
    x=ppp[chosen_features]

    x=x.reset_index(drop=True)
    y=df[(df['Class'] == class1) | (df['Class'] == class2)]['Class']
    y=y.reset_index(drop=True)
    dict={class1:0,class2:1}
    ynew=[ dict[i] for i in y]

    x=np.array(x)
    y=np.array(ynew)
    xtrain,ytrain,xtest,ytest=train_test_split(x, y, 0.2, 3)
    if chosenModel == "Perceptron":
        perceptron_model = Perceptron(2, learningRate, epochsNum, addBias)
        perceptron_model.train(xtrain,ytrain)
        print("Perceptron Accuracy:",perceptron_model.evaluate(xtest, ytest))

    else :
        adaline_model = Adaline(mseThreshold ,2, learningRate, epochsNum )
        adaline_model.train(xtrain,ytrain)
        print("Adaline Accuracy:",adaline_model.evaluate(xtest, ytest))