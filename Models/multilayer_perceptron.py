import math
import numpy as np





class MulilayerPerceptron() :
     
    def __init__( self, hasBias, learning_rate, epochs, layers, activation ) :
        self.hasBias = hasBias
        self.learning_rate = learning_rate
        self.epochs = epochs

        if(activation == 'sigmoid'):
            self.f_dash = self._sigmoid_dash
        elif(activation == 'tanh'):
            self.f_dash = self._tanh_dash
        else:
            raise('please enter a valid activation: [sigmoid, tanh]')

        # the number of weights is equal to the num of neurons in the previous layer
        self.neurons = []
        for i in range(1, len(layers)):
            weightsNum = layers[i-1]
            layerNeuronsNum = layers[i]
            thisLayerNeurons = [None] * layerNeuronsNum
            for j in range(layerNeuronsNum):
                neuron = self._Neuron(self, weightsNum, activation)
                thisLayerNeurons[j] = neuron
            self.neurons.append(thisLayerNeurons)

        print('architecture layers', len(self.neurons))
        for layer in self.neurons:
            print('layer with neuronsNum:', len(layer), 'numOfInputWeights', len(layer[0].weights))

            


    def train(self, input_data, output_data):
        X = np.array(input_data)
        y = np.array(output_data)

        assert len(X) == len(y)
        assert self._isCorrectInputLayer(X)
        classes_num = np.unique(y).size
        assert self._isCorrectOutputLayer(classes_num)

        
        self.classes_num = classes_num

        y_multiClass = self._transform_Y_to_multiclass(y)
        


        sampleSize = X.shape[0]

        error = 0

        for epoch_num in range(self.epochs):

            error = 0

            for i in range(sampleSize):
                layersOutputs = self._forward(X[i])
                # print(layersOutputs)
                error += self._backward(layersOutputs, y_multiClass, i)

                # break # to just make 1 sample of x

            print('epoch', epoch_num, ' accur:', (sampleSize - error), '/', sampleSize)
            if error == 0:
                print('early stopping, error is 0')
                break

        
        print('Training finished with accuracy:', (sampleSize - error) / sampleSize, '\n')
        # self.printWeights()


    def _forward(self, xi):
        previousLayerOutput = xi
        layersOutput = [
            xi
        ]
        for layer in self.neurons:
            neuronsNum = len(layer)
            layersOutput.append(np.zeros(neuronsNum))
            
        
        for layerIndex in range(len(self.neurons)):
            layer = self.neurons[layerIndex]
            
            for neuronIndex in range(len(layer)):
            #    print('epoch', epoch_num, 'i', i, 'layer', layerIndex, 'neuronIndex', neuronIndex)
                previousLayerOutput = layersOutput[layerIndex] # no -1, bec it is 0-based
                layersOutput[layerIndex + 1][neuronIndex] = layer[neuronIndex].forward(previousLayerOutput)
        return layersOutput
                


    def _backward(self, layersOutputs, y_multiClass, sampleIndex):

        layersGradients = []
        
        for layer in self.neurons:
            neuronsNum = len(layer)
            layersGradients.append([0] * neuronsNum)


        last_layer_index = len(self.neurons) - 1
        last_layer = self.neurons[last_layer_index]

        # Last Layer
        for neuronIndex in range(len(last_layer)):
            Y_k_plus1 = layersOutputs[last_layer_index + 1][neuronIndex]
            terminal_gradient = y_multiClass[neuronIndex][sampleIndex] - Y_k_plus1
            error = terminal_gradient

            f_dash = self.f_dash(Y_k_plus1)
            gradient = f_dash * terminal_gradient
            layersGradients[last_layer_index][neuronIndex] = gradient
            neuron = last_layer[neuronIndex]
            neuron.backward(gradient, layersOutputs[last_layer_index])

        # Processing for other layers
        for layerIndex in range(last_layer_index - 1, -1, -1):
            layer = self.neurons[layerIndex]

            for neuronIndex in range(len(layer)):
                terminal_gradient = 0
                nextLayer = self.neurons[layerIndex + 1]

                for nextLayerNeuronIndex in range(len(nextLayer)):
                    weightIndex = neuronIndex + 1 if self.hasBias else neuronIndex
                    weight = nextLayer[nextLayerNeuronIndex].weights[weightIndex]
                    terminal_neuron_gradient = layersGradients[layerIndex + 1][nextLayerNeuronIndex]
                    terminal_gradient += terminal_neuron_gradient * weight

                f_dash = self.f_dash(layersOutputs[layerIndex + 1][neuronIndex])
                gradient = f_dash * terminal_gradient
                layersGradients[layerIndex][neuronIndex] = gradient
                neuron = layer[neuronIndex]
                neuron.backward(gradient, layersOutputs[layerIndex])
                

        last_layer_pred = last_layer_pred = layersOutputs[-1]
        final_class = self._classifyOutput(last_layer_pred)
        error = 1 if y_multiClass[final_class][sampleIndex] != 1 else 0
        return error


    def printWeights(self):
        for i in range(len(self.neurons)):
            layer = self.neurons[i]
            for j in range(len(layer)):
                neuron = layer[j]
                print('neuron', i, j, 'weights', neuron.weights)


    def _sigmoid_dash(self, Y_k_plus1):
        return Y_k_plus1 * (1 - Y_k_plus1)

    def _tanh_dash(self, Y_k_plus1):
        return 1 - (Y_k_plus1 ** 2)



    def _transform_Y_to_multiclass(self, y):
        y_multiClass =  []
        for positiveClassIndex in range(self.classes_num):
            y_class = np.array(y)
            for i in range(len(y)):
                y_class[i] = 1 if y_class[i] == positiveClassIndex else 0
            y_multiClass.append(y_class)
        return y_multiClass


    def _classifyOutput(self, last_layer_pred):
        maxPred = -999
        maxIndex = None
        for i in range(len(last_layer_pred)):
            pred = last_layer_pred[i]
            if pred > maxPred:
                maxPred = pred
                maxIndex = i
        
        return maxIndex
    

    def predict(self, X):
        X = np.array(X)
        sampleSize = X.shape[0]
        y_pred = [0] * sampleSize

        for i in range(sampleSize):
            layersOutputs = self._forward(X[i])
            last_layer_pred = layersOutputs[-1]
            final_class = self._classifyOutput(last_layer_pred)
            y_pred[i] = final_class
        return y_pred

    def _isCorrectInputLayer(self, X):
        firstLayerWeightsExample = self.neurons[0][0].weights
        x_columns_num = X.shape[1]
        if self.hasBias:
            x_columns_num += 1
        return len(firstLayerWeightsExample) == x_columns_num
    
    def _isCorrectOutputLayer(self, classes_num):
        lastLayerNeurons = self.neurons[-1]
        numOfOutputNeurons = len(lastLayerNeurons)
        if classes_num == 2 and numOfOutputNeurons == 1:
            return True
        return len(lastLayerNeurons) == classes_num


    class _Neuron():
        def __init__( self, network_instance, weightsNum, activation) :
            if(activation == 'sigmoid'):
                self.activationFun = self._sigmoid
            elif(activation == 'tanh'):
                self.activationFun = self._tanh
            else:
                raise('please enter a valid activation: [sigmoid, tanh]')
            
            r = np.random.RandomState()
            self.weights = r.random(weightsNum + int(network_instance.hasBias))
            self.network_instance = network_instance
    
    
        def forward(self, x):
            if self.network_instance.hasBias:
                x = self._addBiasToX(x)
                
            ypred = np.dot(x, self.weights)
            activation = self.activationFun(ypred)
            return activation
        
        def backward(self, grdient, x):
            if self.network_instance.hasBias:
                x = self._addBiasToX(x)
    
            self.weights += grdient * self.network_instance.learning_rate * x
    
    
        def _sigmoid(self, z):
            return 1 / (1 + math.exp(-z))
        
        def _tanh(self, z):
            expZ = math.exp(z)
            _expZ = math.exp(-z)
            return (expZ - _expZ) / (expZ + _expZ)
        
        def _addBiasToX(self, x):
            # add a 1 at the begining
            return np.insert(x, 0, 1)

    
    




