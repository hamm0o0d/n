import numpy as np




class MultiLayerPerceptron:
    def __init__(self, num_features, num_layers, num_neurons, num_classes, num_epochs, learn_rate, has_bias, activation):
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.learn_rate = learn_rate
        self.has_bias = has_bias
        self.activation = self._sigmoid if activation == 'sigmoid' else self._tanh
        self.derivative = self._sigmoid_dash if activation == 'sigmoid' else self._tanh_dash

        self.weights = []

        # Add weights for the input layer
        input_size = self.num_features + int(self.has_bias)
        self.weights.append(np.random.rand(self.num_neurons,input_size))

        # Add weights for each hidden layer
        for _ in range(num_layers - 1):
            self.weights.append(np.random.rand(self.num_neurons,self.num_neurons + int(self.has_bias)))

        # Add weights for output layer
        self.weights.append(np.random.rand(self.num_classes,self.num_neurons + int(self.has_bias)))
        # print(int(self.has_bias))
        # for i in self.weights:
        #     print(i.shape)

    

    def forward_propagation(self, input_layer):
        current_layer = input_layer
        layers = []
        # print( self.weights[0].shape,current_layer.shape)
        for i in range(len(self.weights)):
            a = np.dot(self.weights[i],current_layer)
            # print(a.shape)
            l = self.activation(a)
            layers.append(l)
            current_layer = l

        return layers

    def backward_propagation(self, target_output, predicted_outputs):
        gradients = []
        for layer in self.weights:
            gradients.append(np.zeros(self.num_neurons))


        
        output_errors = np.array(target_output) - np.array(predicted_outputs[-1])
        gradients[-1] = (output_errors * self.derivative(predicted_outputs[-1]))
        # print('target_output', target_output)
        # print('output_errors', output_errors)
        # print('predicted_outputs', predicted_outputs[-1])

        


        for i in range(len(self.weights) -2, -1, -1):
            # if i ==0 :break

            next_layer_weights = self.weights[i + 1]
            next_layer_errors = gradients[i + 1]
            current_layer_errors = np.dot(next_layer_weights.T, next_layer_errors )
            current_layer_deltas = current_layer_errors * self.derivative(predicted_outputs[i])
            gradients[i] = current_layer_deltas

        
        return gradients

    def update_weights(self, gradients):
        for i in range(len(self.weights)):
            current_weights = self.weights[i]
            current_errors = gradients[i]

            print('current_weights', current_weights.shape)
            print('current_errors', current_errors.shape)

            # current_errors = current_errors.reshape(-1, 1)
            # print(self.weights[i].shape,current_errors.shape)
            current_weights += self.learn_rate * current_weights * np.array(current_errors)

    def train(self, input_data, target_data):
        for epoch in range(self.num_epochs):
            for i in range(len(input_data)):
                input_layer = input_data[i]
                target_output = target_data[i]

                predicted_output = self.forward_propagation(input_layer)
                # for i in predicted_output: print(i.shape)

                gradients = self.backward_propagation(target_output, predicted_output)

                self.update_weights(gradients)

                break


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def _tanh(self, z):
        expZ = np.exp(z)
        _expZ = np.exp(-z)
        return (expZ - _expZ) / (expZ + _expZ)
        
    def _sigmoid_dash(self, X):
        return X * (1 - X)

    def _tanh_dash(self, X):
        return 1 - (X ** 2)
    

    def predict(self, X):
        X = np.array(X)
        sampleSize = X.shape[0]
        y_pred = np.zeros(sampleSize)

        for i in range(sampleSize):
            layersOutputs = self.forward_propagation(X[i])
            last_layer_pred = layersOutputs[-1]
            # print('last_layer_pred', last_layer_pred)
            final_class = np.argmax(last_layer_pred)
            # print('pred', final_class)
            y_pred[i] = final_class
        return y_pred