from tkinter import *

from run_test import runTest

def test_function():
    addBias = bias_var.get()
    learningRate = float(Lrate_e.get())
    epochs = int(epochs_e.get())
    layers = eval(hidden_layer_e.get())

    runTest(layers, activation.get().lower(), learningRate, epochs, addBias)

# Creating main form and sizing it
master = Tk()
master.geometry('750x500')
master.title('Multilayer Perceptron Task 2')

# Selecting hidden layers GUI
hidden_layer_l = Label(master, text="Enter the layers using the format: [5, 2, 4, 3]")
hidden_layer_l.place(x=10, y=10)

hidden_layer_2 = Label(master, text="Where 5 is the input layer, 2 and 4 are neurons of two hidden layers, 3 is the output neurons of the output layer")
hidden_layer_2.place(x=10, y=40)

hidden_layer_e = Entry(master)
hidden_layer_e.place(x=150, y=75)
hidden_layer_e.insert(0, '[5, 3, 4, 3]')

# Selecting activation function GUI
classes_l = Label(master, text="Select activation function: ")
classes_l.place(x=10, y=110)

activation = StringVar()
activation.set("Sigmoid")
feature1_menu = OptionMenu(master, activation, "Sigmoid", "Tanh")
feature1_menu.place(x=170, y=100)

# Learning rate & number of epochs GUI
Lrate_l = Label(master, text="Learning rate: ")
Lrate_l.place(x=10, y=150)

Lrate_e = Entry(master)
Lrate_e.place(x=110, y=150)
Lrate_e.insert(0, '0.1')

epochs_l = Label(master, text="Number of epochs: ")
epochs_l.place(x=240, y=150)

epochs_e = Entry(master)
epochs_e.place(x=350, y=150)
epochs_e.insert(0, '1000')

# Create a variable to hold the bias checkbox state
bias_var = IntVar(value=1)

# Create a checkbox
checkbox = Checkbutton(master, text="Bias", variable=bias_var)
checkbox.place(x=500, y=150)

# Test button
button = Button(master, text="Train-Test", width=10, height=3, command=test_function)
button.place(x=250, y=200)

mainloop()
