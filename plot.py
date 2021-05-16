import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import numpy as np


def PlotConfusionMatrix(cm, Target_Names=None, Normalize=False, FileName=None):
    """
    cm:         : confusion matrix
    Target_Name : array-like with class names ['a','b','c'] or ohe.categories_[0]
    Normalize   : If False, plot the raw numbers
                  If True, plot the proportions
    FileName    : file name to save current figure in JPG format
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    
    if Normalize:
        # normalize the confusion matrix data by dividing its values over sum of rows
        cm = cm.astype('float') / cm.sum(axis=1)
        threshold = cm.max() / 1.5
    else:
        threshold = cm.max() / 2
    
    if Target_Names is not None:
        tricks = np.arange(len(Target_Names))
        plt.xticks(tricks, Target_Names)
        plt.yticks(tricks, Target_Names)

    for x, y in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if Normalize:
            plt.text(y, x, "{:.4f}".format(cm[x,y]),
                     horizontalalignment="center", color="white" if cm[x,y] > threshold else "black")
        else: 
            plt.text(y,x, "{0}".format(cm[x, y]),
                     horizontalalignment="center", color="white" if cm[x,y] > threshold else "black")

    plt.title('Confusion Matrix', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)
    if FileName is not None:
        plt.savefig(FileName,format="jpg")
    plt.show()


def PlotPrediction(trained_model, encoder, entry):
    predictions_df = pd.DataFrame(
        zip(encoder.categories_[0], trained_model.predict(entry)[0]),
        columns=['Target', 'Softmax'])

    with plt.style.context('seaborn-talk'):
        fig, ax = plt.subplots(figsize=(6,3))
        predictions_df['Softmax'].plot(kind='barh')
        ax.set(title='Prediction Probabilities')

    # fig, ax = plt.subplots(figsize=(6,3))
    # predictions_df['Softmax'].plot(kind='barh')
    # g = sns.barplot(x='Softmax', y='Target', data= predictions_df)
    # for idx, row in predictions_df.iterrows():
    #     g.text(x=row.Softmax+.05,y=row.name, s=round(row.Softmax,3),color='k', va='center',  ha="right", fontsize=8)
    # g.set_xlim(-0.03, 1.05)
    # return predictions_df

def Plot_History(history):
    # if not isinstance(history, keras.callbacks.History):
    #     return f"{history} is not a valid keras.callbacks.History"
    metrics = [v for v in history.history.keys() if not v.startswith('val')]
    
    fig, ax=plt.subplots(1, len(metrics), figsize=(20,5))
    
    for n, metric in enumerate(metrics):
        plt.subplot(1,len(metrics),n+1)
        plt.plot(history.epoch, 
                 history.history[metric],
                 color=colors[0],
                 linestyle="-",
                 linewidth='3',
                 label='Training')
        
        val_metric = f"val_{metric}"
        if val_metric in history.history.keys():
            plt.plot(history.epoch,
                     history.history[val_metric],
                     color=colors[1],
                     linestyle="-",
                     linewidth='3',
                     label='Validation')
        
        plt.xlabel('EPOCH', fontsize=18)
        plt.ylabel(metric.upper(), fontsize=18)
        plt.subplots_adjust(wspace=.15, hspace=.2)

        fig.tight_layout()
        plt.legend()
