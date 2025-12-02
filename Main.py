import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import random
from utils import vector_infos, make_data, plot_data, plot_results

"""
Main Script for Logistic Regression Example
"""    

#========================== Parameters Config 
data_size = 1000 
data_cols = 2 
data_noise = 0.2
random_seed = 123
        
random.seed(random_seed)
np.random.seed(random_seed)


#========================== Generate Data
x, y = make_data(data_size, data_noise)

#------------ Visualize Data
plot_data(x, y, legend=True,title="Data Distribution")
vector_infos("Feature X", x)
vector_infos("Labels y", y)


#------------ Prepare data for training and testing
split_ratio = 0.8
n = int(data_size*split_ratio)

x_train, x_test = x[:n], x[n:]
y_train, y_test = y[:n], y[n:]

plot_data(x_test, y_test, colors=('gray', 'gray'), legend=True, title="Data to classify", fig_name="data_to_classify.png")

#------------ Normalization
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

#------------ Log info. 

vector_infos("Training Feature X_train", x_train)
vector_infos("Training Labels y_train", y_train)
vector_infos("Testing Feature X_test", x_test)
vector_infos("Testing Labels y_test", y_test)


# ============ Training ============ 
""" 
We can choose different solvers for optimization:
'liblinear', 'newton-cg', 'sag', 'saga' and 'lbfgs'
"""
logreg = LogisticRegression(C=1000, verbose=0, solver='saga')
#------------ Fit the data.
logreg.fit(x_train, y_train)

#------------ Predict
y_pred = logreg.predict(x_test)

#------------ Evaluate
plot_results(x_test, y_test, y_pred, fig_name="results_train_plot.png")


#============ Enhance data ============

x_train_enhanced = np.c_[x_train,
                         x_train[:, 0] ** 2,
                         x_train[:, 1] ** 2,
                         x_train[:, 0] ** 3,
                         x_train[:, 1] ** 3]

x_test_enhanced = np.c_[x_test,
                        x_test[:, 0] ** 2,
                        x_test[:, 1] ** 2,
                        x_test[:, 0] ** 3,
                        x_test[:, 1] ** 3]


logreg = LogisticRegression(C=1e5, verbose=0, solver='saga', max_iter=5000, n_jobs=-1)

#------------ Fit the data.
logreg.fit(x_train_enhanced, y_train)

#------------ Predict
y_pred = logreg.predict(x_test_enhanced)
#------------ Evaluate
plot_results(x_test_enhanced, y_test, y_pred, fig_name="results_train_plot_enhanced.png")
