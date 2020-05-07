
# Plot accuracy/epochs and loss/epochs for testing and training
def plot(filename, stats):
    import matplotlib.pyplot as plt
    # Plot training accuracy and loss by epoch
    plt.plot(stats.epochs, stats.accuracy, label='Accuracy')
    plt.plot(stats.epochs, stats.loss, label='Loss')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend(loc='upper left')
    plt.savefig('./plots/' + filename + '_training')
    plt.clf()
    # Plot testing accuracy and loss by epoch
    plt.plot(stats.epochs, stats.val_accuracy, label='Accuracy')
    plt.plot(stats.epochs, stats.val_loss, label='Loss')
    plt.title('Testing Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend(loc='upper left')
    plt.savefig('./plots/' + filename + '_testing')
    plt.clf()
    # Plot training and testing accuracy and loss by epoch
    plt.plot(stats.epochs, stats.val_accuracy, label='Val Accuracy')
    plt.plot(stats.epochs, stats.val_loss, label='Val Loss')
    plt.plot(stats.epochs, stats.accuracy, label='Accuracy')
    plt.plot(stats.epochs, stats.loss, label='Loss')
    plt.title('Metrics by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend(loc='upper left')
    plt.savefig('./plots/' + filename)
    plt.clf()

# Saves training statistics in csv
def stats(filename, stats):
    import numpy as np
    agg_stats = np.asarray([stats.epochs, stats.loss, stats.accuracy, stats.val_loss, stats.val_accuracy]).T
    np.savetxt('./stats/' + filename + '.csv', agg_stats, header='epochs,loss,accuracy,val_loss,val_accuracy', delimiter=',')

# Saves model summary in csv
def summary(filename, stats, total_epochs, model, y_test, x_test):
    from sklearn.metrics import classification_report
    import numpy as np
    y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
    y_pred = model.predict_classes(x_test)
    confusion = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, zero_division=0))

    filew = open('summary.csv', 'a')
    filew.write('\n' +
                filename + ',' + 
                str(total_epochs) + ',' + 
                "{:.2f}".format(stats.epoch_time) + ',' + 
                "{:.2f}".format(stats.epoch_time * total_epochs / 60) + ',' + 
                str(stats.m_epoch) + ',' + 
                "{:.4f}".format(stats.m_accuracy) + ',' + 
                "{:.4f}".format(stats.m_loss) + ',' + 
                "{:.3f}".format(confusion['0']['precision']) + ',' + "{:.3f}".format(confusion['0']['recall']) + ',' +
                "{:.3f}".format(confusion['0']['f1-score']) + ',' + str(confusion['0']['support']) + ',' +
                "{:.3f}".format(confusion['1']['precision']) + ',' + "{:.3f}".format(confusion['1']['recall']) + ',' +
                "{:.3f}".format(confusion['1']['f1-score']) + ',' + str(confusion['1']['support']) + ',' +
                "{:.3f}".format(confusion['2']['precision']) + ',' + "{:.3f}".format(confusion['2']['recall']) + ',' +
                "{:.3f}".format(confusion['2']['f1-score']) + ',' + str(confusion['2']['support']) + ',' +
                "{:.3f}".format(confusion['3']['precision']) + ',' + "{:.3f}".format(confusion['3']['recall']) + ',' +
                "{:.3f}".format(confusion['3']['f1-score']) + ',' + str(confusion['3']['support']) + ',' +
                "{:.3f}".format(confusion['4']['precision']) + ',' + "{:.3f}".format(confusion['4']['recall']) + ',' +
                "{:.3f}".format(confusion['4']['f1-score']) + ',' + str(confusion['4']['support'])
            )
    filew.close()
