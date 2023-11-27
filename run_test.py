
import numpy as np
from Models.perceptron import Perceptron
from Models.adaline import Adaline
from Models.multilayer_perceptron import MultiLayerPerceptron
from HelperFunctions.train_test_split import train_test_split
from HelperFunctions.load_dataset import load_dataset
from HelperFunctions.result_plotter import plotClassifier


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
    model = None
    if chosenModel == "Perceptron":
        model = Perceptron(2, learningRate, epochsNum, addBias)
        model.train(xtrain,ytrain)
        print("Perceptron Accuracy:",model.evaluate(xtest, ytest))

    elif chosenModel == "Adaline" :
        model = Adaline(mseThreshold ,2, learningRate, epochsNum )
        model.train(xtrain,ytrain)
        print("Adaline Accuracy:",model.evaluate(xtest, ytest))
    
    else:
        model = MultiLayerPerceptron(
            num_features=5,
            num_layers=2,
            num_neurons=3,
            num_classes=3,
            learn_rate=0.01,
            has_bias=False,
            activation='sigmoid'
        )

        x = df.drop(columns=['Class'])
        y = df['Class']

        print(x.shape)

        xtrain,ytrain,xtest,ytest=train_test_split(x, y, 0.4, 3)

        model.train(xtrain,ytrain)
        print("MLP Accuracy:",model.evaluate(xtest, ytest))
    
    plotClassifier(xtest, ytest, model, class1, class2, chosen_features[0], chosen_features[1])