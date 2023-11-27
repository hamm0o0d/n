
from HelperFunctions.load_dataset import load_dataset
from HelperFunctions.label_encoder import LabelEncoder
from HelperFunctions.on_hot_encoder import one_hot_encode
from HelperFunctions.train_test_split import train_test_split
from Models.multilayer_perceptron import MultiLayerPerceptron
import ui_screen


# ui_screen.openGUIScreen()

model = MultiLayerPerceptron(
            num_features=5,
            num_layers=2,
            num_neurons=3,
            num_classes=3,
            learn_rate=0.01,
            num_epochs=1,
            has_bias=False,
            activation='sigmoid'
        )


df = load_dataset()

encoder = LabelEncoder(df['Class'])
df['Class'] = df['Class'].apply(lambda x: encoder.encode(x))

y = df['Class']
x = df.drop(columns=['Class'])



xtrain,ytrain,xtest,ytest=train_test_split(x, y, 0.4, 3)

target_data_encoded = one_hot_encode(ytrain, 3)
print(target_data_encoded.shape)
model.train(xtrain, target_data_encoded)

y_pred = model.predict(xtest)
correct = 0
import numpy as np
y_test = np.array(ytest)

for i in range(len(y_test)):
    # print('y_test', y_test[i], 'pred', y_pred[i])
    if(y_test[i] == y_pred[i]):
        correct += 1
print('accur', correct/len(y_test))