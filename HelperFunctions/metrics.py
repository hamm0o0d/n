
def unique_elements(list1,list2):
    unique = []

    for x in list1:

        if x not in unique:
            unique.append(x)
    for x in list2:

        if x not in unique:
            unique.append(x)
    return unique


def confusion_matrix(allActual, allPredicted, unique ):

    class_a = unique[0]
    class_b = unique[1]
    trueA = 0
    falseA = 0
    trueB = 0
    falseB = 0
    for actual , predicted in  zip(allActual , allPredicted):
        # case one: true class_a
        if predicted == class_a and actual==class_a:
            trueA+=1
        # case two: false class_a
        elif predicted == class_a and actual != class_a:
            falseA+=1
        # case three: true class_b
        if predicted == class_b and actual == class_b:
            trueB += 1
        # case two: false class_b
        elif predicted == class_b and actual != class_b:
            falseB += 1
    acurracy = (trueA + trueB) / len(allActual)
    print("___________________________________________\n")
    print("                confusion matrix           \n")
    print("___________________________________________\n")
    print(" true class-one: ", trueA, "  ", "false class-one: ", falseA, "\n")
    print(" true class-two: ", trueB, "  ", "false class-two: ", falseB, "\n")
    print("accuracy = ", acurracy)
    print("___________________________________________\n")



