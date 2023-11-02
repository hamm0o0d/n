from tkinter import *
from run_test import runTest

def openGUIScreen():


    def test():
        addBias = bias_var.get()
        selectedModel = selected_model.get()
        chosenFeatures = [feature1_var.get(), feature2_var.get()]
        class1 = class1_var.get()
        class2 = class2_var.get()
        learningRate = float(Lrate_e.get())
        epochs = int(epochs_e.get())
        mseThreshold = 0
        if(selectedModel == 'Adaline'):
            mseThreshold = float(mse_threshold_e.get())

        runTest(selectedModel, chosenFeatures, class1, class2, learningRate, epochs, mseThreshold, addBias)




    # creating main form and sizing it
    master = Tk()
    master.geometry('750x500')

    # selecting 2 feature GUI

    features_l=Label(master, text="Select 2 Features: ")
    features_l.place(x=10,y=10)

    feature1_var= StringVar()
    feature1_var.set("Area")
    feature1_menu=OptionMenu(master,feature1_var, "Area","Perimeter","MajorAxisLength","MinorAxisLength","roundnes")
    feature1_menu.place(x=110,y=10)

    feature2_var= StringVar()
    feature2_var.set("Perimeter")
    feature2_menu=OptionMenu(master,feature2_var, "Area","Perimeter","MajorAxisLength","MinorAxisLength","roundnes")
    feature2_menu.place(x=250,y=10)


    # selecting 2 Classes GUI

    classes_l=Label(master, text="Select 2 Classes: ")
    classes_l.place(x=10,y=50)

    class1_var= StringVar()
    class1_var.set("BOMBAY")
    class1_menu=OptionMenu(master,class1_var, "BOMBAY","CALI","SIRA")
    class1_menu.place(x=110,y=50)

    class2_var= StringVar()
    class2_var.set("CALI")
    class2_menu=OptionMenu(master,class2_var, "BOMBAY","CALI","SIRA")
    class2_menu.place(x=250,y=50)


    # learning rate & no of epochs
    Lrate_l=Label(master, text="learning rate: ")
    Lrate_l.place(x=10,y=100)

    Lrate_e=Entry(master)
    Lrate_e.place(x=110,y=100)

    epochs_l=Label(master, text="no of epochs: ")
    epochs_l.place(x=240,y=100)


    epochs_e=Entry(master)
    epochs_e.place(x=330,y=100)


    # MSE threshold
    mse_threshold_l=Label(master, text="MSE threshold: ")
    mse_threshold_l.place(x=430,y=100)

    mse_threshold_e=Entry(master)
    mse_threshold_e.place(x=520,y=100)

    # rdio button to choose between perceptron and adaline
    selected_model = StringVar(value='Perceptron')
    Perceptron_btn = Radiobutton(master, text="Perceptron", variable=selected_model, value="Perceptron")
    Adaline_btn = Radiobutton(master, text="Adaline", variable=selected_model, value="Adaline")
    Perceptron_btn.place(x=10,y=120)
    Adaline_btn.place(x=10,y=150)



    # Create a variable to hold the bias checkbox state
    bias_var = BooleanVar(value=True)

    # Create a checkbox
    checkbox = Checkbutton(master, text="Bias", variable=bias_var)
    checkbox.place(x=650,y=100)

    # test buttons
    button= Button(master, text="test", command = test)
    button.place(x=50,y=200)


    mainloop()