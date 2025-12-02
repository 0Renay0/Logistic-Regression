import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import random

# Funcions from Fidle
def vector_infos(name, V):
    # Print basic information about a vector
    with np.printoptions(precision=4, suppress=True):
        print(f"{name}:")
        print(f"  Shape: {V.shape}")
        print(f"  Dtype: {V.dtype}")
        print(f"  Min: {V.min()}")
        print(f"  Max: {V.max()}")
        print(f"  Mean: {V.mean()}")
        print(f"  Std: {V.std()}")
        print()
        
        
def Decision(work,sleep):
    # return 1 if success, 0 if fail depending on work and sleep hours 
    work_min = 4
    sleep_min = 5
    game_max = 3
    
    if sleep < sleep_min:
        return 0  # fail ! 
    if work < work_min:
        return 0  # fail !
    
    game = 24 - (work + sleep) + random.gauss(0,0.5)
    if game > game_max:
        return 0  # fail
    
    return 1  # success !


def make_data(size,noise):
    # Generate dataset of given size of people with work and sleep hours
    x = []
    y = []
    for i in range (size):
        work = random.gauss(5,1)  # average 5 hours of work
        sleep = random.gauss(7,1.5) 
        r = Decision(work,sleep)
        x.append([work, sleep])
        y.append(r)
    return np.array(x), np.array(y)

def plot_data(x, y, colors=('green', 'red'), legend = True):
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(10,8)
    
    ax.plot(x[y==1, 0], x[y==1, 1], 'o', color=colors[0], markersize=4, label='Success')
    ax.plot(x[y==0, 0], x[y==0, 1], 'o', color=colors[1], markersize=4, label='Fail')

    if legend:
        ax.legend()
        
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.xlabel("Work hours")
    plt.ylabel("Sleep hours")
    plt.show()
    
    
    def plot_results(x_test, y_test, y_pred):
        
        Precision = metrics.precision_score(y_test, y_pred)
        Recall = metrics.recall_score(y_test, y_pred)
        
        print(f"Precision: {Precision:.4f}")
        print(f"Recall: {Recall:.4f}")
        
        x_pred_pos = x_test[y_pred == 1]
        x_real_pos = x_test[y_test == 1]
        x_pred_neg = x_test[y_pred == 0]
        x_real_neg = x_test[y_test == 0]
        
        fig, axs = plt.subplots(2,2)
        fig.subplots_adjust(wspace=.1, hspace=.2)
        fig.set_size_inches(14,10)
        
        # Predicted Vs Real positive
        axs[0,0].plot(x_pred_pos[:, 0], x_pred_pos[:, 1], 'o', markersize=4, label='Predicted Positive')
        axs[0,0].plot(x_real_pos[:, 0], x_real_pos[:, 1], 'x', markersize=4, label='Real Positive')