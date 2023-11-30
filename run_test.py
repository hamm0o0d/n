

from HelperFunctions.label_encoder import LabelEncoder
from Models.multilayer_perceptron import MulilayerPerceptron
from HelperFunctions.train_test_split import train_test_split
from HelperFunctions.load_dataset import load_dataset
from HelperFunctions.metrics import confusion_matrix_multiclass


def runTest(layers, activation, learningRate, epochsNum, addBias):

    df = load_dataset()

    class_encoder = LabelEncoder(df['Class'])
    df['Class'] = df['Class'].apply(lambda x: class_encoder.encode(x))
    
    y = df['Class']
    x = df.drop(columns=['Class'])
    
    
    
    xtrain,ytrain,xtest,ytest=train_test_split(x, y, 0.4, 3)

    model = MulilayerPerceptron(hasBias=addBias, learning_rate=learningRate, epochs=epochsNum, activation=activation, layers=layers)
    
    model.train(xtrain, ytrain)
    
    y_pred = model.predict(xtest)
    
    confusion_matrix_multiclass(y_pred, ytest, class_encoder.getLabels())

