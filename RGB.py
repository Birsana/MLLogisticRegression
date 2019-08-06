# This is my first machine learning algorithm I've coded, using logistic regression. It will pick whether a light or
# dark background is better for a certain text colour based on its RGB values

import numpy as np
from matplotlib import pyplot as plt

examiningCode = False  # change to true if you want to run the program from a debugging/testing point of view

# I manually collected 35 examples, by randomly picking RGB values for text, amd comparing them on a white and dark
# background

# Just as a side note, while collecting data, some colours looked
# arguablly equally good on both light and dark backgrounds, but I used my own judgement to pick whether a light or dark
# background was the better choice.

# This is a small amount of data, but this RGB background picker is a simple model, so I still expect the computer's
# predictions to be fairly accurate. Of course, getting more data would be better.

data = np.array([[0, 0, 102, 1],
                 [255, 51, 153, 0],
                 [0, 255, 0, 0],
                 [204, 153, 0, 0],
                 [51, 51, 255, 0],
                 [51, 153, 255, 0],
                 [128, 0, 0, 1],
                 [51, 51, 0, 1],
                 [0, 255, 153, 0],
                 [255, 153, 255, 0],
                 [102, 0, 204, 1],
                 [153, 0, 51, 1],
                 [77, 102, 0, 1],
                 [255, 228, 204, 0],
                 [102, 48, 0, 1],
                 [230, 255, 243, 0],
                 [199, 23, 144, 0],
                 [11, 24, 5, 1],
                 [156, 23, 2, 1],
                 [36, 65, 40, 1],
                 [7, 77, 12, 1],
                 [230, 18, 199, 0],
                 [47, 79, 79, 1],
                 [0, 128, 128, 1],
                 [224, 255, 255, 0],
                 [216, 191, 216, 0],
                 [128, 0, 128, 1],
                 [255, 105, 180, 0],
                 [255, 228, 196, 0],
                 [245, 222, 179, 0],
                 [85, 107, 47, 1],
                 [139, 0, 139, 1],
                 [160, 82, 45, 0],
                 [0, 0, 139, 1],
                 [75, 0, 130, 1]
                 ])

# 0 represents a dark background, while 1 represents a white background, the first 3 columns are RGB values

# Get the RGB values + answer individually from the data
rValue = data[:, 0]
gValue = data[:, 1]
bValue = data[:, 2]
classification = data[:, 3]

m = len(classification)

y = classification[:, np.newaxis]  # turn from array into vector


# Now we'll plot our data to see what we can extrapolate from it
# Plot each RGB component individually, to see if we can notice any pattern

def testGraphs():
    plt.scatter(rValue, classification)
    plt.xlabel('R Values')
    plt.ylabel('Background')
    plt.show()

    plt.scatter(gValue, classification)
    plt.xlabel('G Values')
    plt.ylabel('Background')
    plt.show()

    plt.scatter(bValue, classification)
    plt.xlabel('B Values')
    plt.ylabel('Background')
    plt.show()
    return


if examiningCode:
    testGraphs()
# After creating the scatter plots, an exact correlation between each individual RGB value and the best background is
# still unknown. I'll try creating a new variable called rgbSum, which is the total sum of a colour's RGB values,
# and graph that

justRGB = data[:, :3]
rgbSum = np.zeros(np.size(justRGB, 0))

for i in range(len(rgbSum)):  # feature scaling below to get all of our data approximately between -1 and 1.
    rgbSum[i] = (np.sum(justRGB[i]) - 340) / 734  # 340 is the approximate mean, and 734 is the approximate range


# mean = rgbSum.mean()
# range = rgbSum.max() #not actually the correct range but close enough


# Now I'll try plotting the RGB sum
def testRGB():
    plt.scatter(rgbSum, classification)
    plt.xlabel('RGB Values')
    plt.ylabel('Background')
    plt.show()
    return


if examiningCode:
    testRGB()
# There seems to be a fairly strong correlation between the sum of the RGB values and the desired background:
# the higher the sum, the more likely the background is to be dark


# I'll be using the RGB sum as the sole feature for this logistic regression model.
temp = np.ones(np.size(rgbSum))

X = np.column_stack((temp, rgbSum))  # our data matrix, the first column contains ones by convention

theta = np.random.randn(2, 1)  # parameter matrix, only has two parameters because we have two features


# if examiningCode:
#   print(theta)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Cost Function:
costs = []


def costFunction(theta, X, y):
    J = (-1 / m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta)))
                          + np.multiply((1 - y), np.log(1 - sigmoid(X @ theta))))
    return J


# Gradient Descent function
def gradientDescent(theta, X, y, lRate, iterations):
    for i in range(iterations):

        gradient = ((1 / m) * X.T @ (sigmoid(X @ theta) - y))

        theta = theta - lRate * gradient
        if i % 100 == 0:
            costs.append(costFunction(theta, X, y))
    return theta


'''IMPORTANT SIDE NOTE
normally, a data set would be divided into a training, cross validation, and test set. However, due to the fact that
I didn't collect too much data and I mostly wanted to focus on just programming logistic regression properly, I
decided to use all my data for the training set. Not using a cross validation set means it is hard to tell if I have a
 high variance or high bias. However, due to the fact that I'm only using one feature, I would assume I would most
likely be suffering from some bias. Also, no test set means I can't give a generalized cost/accuracy for this algorithm,
but again I was mainly just focused on getting logistic regression to work. I also did not include regurlization because
with just one feature I don't think overfitting would be a problem '''

optimalTheta = gradientDescent(theta, X, y, 0.1,
                               100000)  # chose arbitrary learing rate and iteration, in practice should test
# different values

if examiningCode:
    plt.plot(costs)
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.show()

if examiningCode:
    print(optimalTheta)

# will plot decision boundary, find where h(z) = 0.5, essentially theta0 + theta1*x = 0
xDecision = -optimalTheta[0] / optimalTheta[1]


# graph with decision boundary
def decisionBound():
    plt.scatter(rgbSum, classification)
    plt.xlabel('RGB Values')
    plt.ylabel('Background')
    plt.axvline(x=xDecision, c='r')
    plt.show()
    return


if examiningCode:
    decisionBound()
# User inputs their own rgb value
if (not examiningCode):
    rUser = int(input("Enter an R Value for your text"))
    gUser = int(input("Enter an G Value for your text"))
    bUser = int(input("Enter an B Value for your text"))


    while rUser > 255 or gUser > 255 or bUser > 255 or rUser < 0 or gUser < 0 or bUser < 0:
        print("Please enter values between 0 and 255")
        rUser = int(input("Enter an R Value for your text"))
        gUser = int(input("Enter an G Value for your text"))
        bUser = int(input("Enter an B Value for your text"))
    darkBackground = False
    lightBackground = False
    rgbUser = rUser + gUser + bUser

    prediction = sigmoid(optimalTheta[0] + optimalTheta[1] * (rgbUser - 340) / 734)

    bckg = ""
    if prediction <= 0.5:
        bckg = "Black"
    else:
        bckg = "White"

    outputFile = open('textBackground.html', 'w')

    text = """<html>
    <head></head>
    <body><p style = "color:rgb(%s, %s, %s); background-color:%s">

    This text looks best on this background</p></body>
    </html>
    """ % (rgbUser, gUser, bUser, bckg)
    outputFile.write(text)
    outputFile.close()

    print("Please open the file textBackground.html")

'''Final comments:

If I were to do this problem again, I'd try to have better and more methodotical data collecting process than eyeballing text colour on a 
background, and collect far more data so I could have a training set, cross validation set, and test set. I'd also
try adding more features, like maybe the individual R G B values; even though when I plotted them I saw no correlation,
there may have actaully been one. This would also let me implement regularization. I'd also choose a better way to pick
my learning rate and number of iterations'''


