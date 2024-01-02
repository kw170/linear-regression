import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

# features (G1, G2, studytime, failures, absences)

x = np.array(data.drop(predict, axis=1))

# label (G3)
y = np.array(data[predict])

# x and y train data is used to train the model
# x and y test data is reserved for testing the model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()

# loop used to find and save most accurate model
""""
best = 0
for i in range(10000):
    # repeated line to randomize data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    
    # creates best fit line using x and y training data
    linear.fit(x_train, y_train)

    # determines accuracy of prediction using coefficient of determination (R^2)
    acc = linear.score(x_test, y_test)

    # updates model if more accurate
    if acc > best:
        print(acc)
        best = acc
        # saves pickle file in directory
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

# load model from pickle file into linear
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# y = m1x1 + m2x2 + ... + (mn)(xn) + b
print("Co: ", linear.coef_)
print("Y - Intercept / b: ", linear.intercept_)

predictions = linear.predict(x_test)

# prints predicted g3, input data, and actual g3
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'G2'
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()