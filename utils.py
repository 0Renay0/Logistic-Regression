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
    
    game = 24 - 10 -(work + sleep) + random.gauss(0,0.4)
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

def plot_data(x, y, colors=('green', 'red'), legend = True, title="Data Distribution", fig_name="data_plot.png"):
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(10,8)
    
    ax.plot(x[y==1, 0], x[y==1, 1], 'o', color=colors[0], markersize=4, label='Success')
    ax.plot(x[y==0, 0], x[y==0, 1], 'o', color=colors[1], markersize=4, label='Fail')

    if legend:
        ax.legend()
        
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.xlabel("Work hours")
    plt.ylabel("Sleep hours")
    if title:
        plt.title(title)
    if fig_name:
        plt.savefig(fig_name, dpi=140)
    plt.show()    
    
def plot_results(x_test,y_test, y_pred, fig_name="results_train_plot.png"):
    '''Affiche un resultat'''

    precision = metrics.precision_score(y_test, y_pred)
    recall    = metrics.recall_score(y_test, y_pred)

    print("Accuracy = {:5.3f}    Recall = {:5.3f}".format(precision, recall))

    x_pred_positives = x_test[ y_pred == 1 ]     # items prédits    positifs
    x_real_positives = x_test[ y_test == 1 ]     # items réellement positifs
    x_pred_negatives = x_test[ y_pred == 0 ]     # items prédits    négatifs
    x_real_negatives = x_test[ y_test == 0 ]     # items réellement négatifs

    
    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(wspace=.1,hspace=0.2)
    fig.set_size_inches(14,10)
    
    axs[0,0].plot(x_pred_positives[:,0], x_pred_positives[:,1], 'o',color='lightgreen', markersize=10, label="Prédits positifs")
    axs[0,0].plot(x_real_positives[:,0], x_real_positives[:,1], 'o',color='green',      markersize=4,  label="Réels positifs")
    axs[0,0].legend()
    axs[0,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axs[0,0].set_xlabel('$x_1$')
    axs[0,0].set_ylabel('$x_2$')


    axs[0,1].plot(x_pred_negatives[:,0], x_pred_negatives[:,1], 'o',color='lightsalmon', markersize=10, label="Prédits négatifs")
    axs[0,1].plot(x_real_negatives[:,0], x_real_negatives[:,1], 'o',color='red',        markersize=4,  label="Réels négatifs")
    axs[0,1].legend()
    axs[0,1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axs[0,1].set_xlabel('$x_1$')
    axs[0,1].set_ylabel('$x_2$')
    
    axs[1,0].plot(x_pred_positives[:,0], x_pred_positives[:,1], 'o',color='lightgreen', markersize=10, label="Prédits positifs")
    axs[1,0].plot(x_pred_negatives[:,0], x_pred_negatives[:,1], 'o',color='lightsalmon', markersize=10, label="Prédits négatifs")
    axs[1,0].plot(x_real_positives[:,0], x_real_positives[:,1], 'o',color='green',      markersize=4,  label="Réels positifs")
    axs[1,0].plot(x_real_negatives[:,0], x_real_negatives[:,1], 'o',color='red',        markersize=4,  label="Réels négatifs")
    axs[1,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axs[1,0].set_xlabel('$x_1$')
    axs[1,0].set_ylabel('$x_2$')

    axs[1,1].pie([precision,1-precision], explode=[0,0.1], labels=["","Errors"], 
                 autopct='%1.1f%%', shadow=False, startangle=70, colors=["lightsteelblue","coral"])
    axs[1,1].axis('equal')
    
    if fig_name:
        plt.savefig(fig_name, dpi=140)
    plt.show()
        
  