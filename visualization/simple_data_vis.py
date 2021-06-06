"""imports"""
import numpy as np
import matplotlib.pyplot as plt
import utils
from surrogate_models.nn_defs import multi_mse

def histograms(data,x_data,y_data):
    """builds histograms"""
    n_outputs = len(y_data)
    n_inputs = len(x_data)
    n_gen = int(np.ceil(n_inputs**0.5))

    fig, axis = plt.subplots(n_gen, n_gen, figsize = (12,12))
    fig.tight_layout()

    for a_n,var in zip(axis.flat,x_data):
        data[var].hist(axis = a_n)
        a_n.set_title(var)

    fig, axis = plt.subplots(n_outputs, 1, figsize = (3,6))
    fig.tight_layout()

    for a_n,var in zip(axis.flat,y_data):
        data[var].hist(axis = a_n)
        a_n.set_title(var)

    return fig,axis

def scatter_hats(models, train, test = None, settings = {}, n_points = 50, display_info = True):
    """main function for visualizing model prediction v. true value complete with mse"""
    y_data = train[1].columns
    markers = ['.','^'] # ok for two outputs only

    sample_inputs_train = train[0].sample(n_points)
    sample_outputs_train = train[1].loc[sample_inputs_train.index]
    if test is not None: sample_inputs_test = test[0].sample(n_points)
    if test is not None: sample_outputs_test = test[1].loc[sample_inputs_test.index]

    f = plt.figure(figsize = (10*len(models)/3,10*len(models)))
    for i,mod in enumerate(models):
        model_name = utils.get_model_name(mod, settings['dataset'])
        plt.subplot(1,len(models),i+1)

        train_hat = mod(sample_inputs_train.to_numpy())
        if test is not None: test_hat = mod(sample_inputs_test.to_numpy())

        for j in range(len(y_data)):
            try:
                plt.errorbar(sample_outputs_train.to_numpy()[:,j],
                        train_hat[j].numpy(),
                            marker = markers[j], color = 'k', alpha = 0.5,
                            ls = 'none',  label = y_data[j])
                if test is not None: plt.errorbar(sample_outputs_test.to_numpy()[:,j],
                        test_hat[j].numpy(),
                            marker = markers[j], color = 'r', alpha = 0.5,
                            ls = 'none')
            except:
                plt.errorbar(sample_outputs_train.to_numpy()[:,j], train_hat[j].mean().numpy(),
                        yerr = train_hat[j].stddev().numpy().squeeze(),
                            marker = markers[j], color = 'k', alpha = 0.5, ls = 'none',
                            label = y_data[j])
                if test is not None: plt.errorbar(sample_outputs_test.to_numpy()[:,j], test_hat[j].mean().numpy(),
                        yerr = test_hat[j].stddev().numpy().squeeze(),
                            marker = markers[j], color = 'r', alpha = 0.5, ls = 'none')

        plt.title(model_name + ' ({} params)'.format(utils.count_parameters(mod)))
        xmax = 1.1
        plt.gca().set_aspect('equal')
        plt.xlim([0,xmax])
        plt.ylim([0,xmax])
        plt.plot([0,xmax],[0,xmax],'k',alpha=0.25)
        plt.xlabel('true')
        plt.ylabel('predicted')
        if i == 0:
            plt.legend(frameon = False)

        rmse = multi_mse(train[1], mod(train[0].to_numpy()))
        plt.text(0.5,0.12,f"Train MSE: {round(np.sum(rmse)*100, 2)}", color = 'k')
        if test is not None:
            rmse = multi_mse(test[1], mod(test[0].to_numpy()))
            plt.text(0.5,0.04,f"Test MSE: {round(np.sum(rmse)*100, 2)}", color = 'r')

    plt.tight_layout()
    if display_info:
        plt.text(2, .80, '.', fontsize=1)
        # could be a loop
        plt.text(1.20, .90, 'Dataset: '+settings['dataset'], fontsize=12)
        plt.text(1.20, .70, 'Optimizer: '+settings['optimizer'], fontsize=12)
        plt.text(1.20, .50, 'Learning Rate: '+str(settings['learning_rate']), fontsize=12)
        plt.text(1.20, .30, 'Loss Weights: '+str(settings['loss_weights']), fontsize=12)
        plt.text(1.20, .10, 'Epochs: '+str(settings['epochs']), fontsize=12)
    return f
    
def training_curves(models, y_data, settings, histories, smoothing = 1):
    """function for building training curves"""
    epochs = np.arange(settings['epochs'])
    markers = ['.','^'] # ok for two outputs only
    plt.figure(figsize = (10*len(models)/3,10*len(models)))
    for i,mod in enumerate(models):
        model_name = utils.get_model_name(mod, settings['dataset'])
        plt.subplot(1,len(models),i+1)
        for j in range(len(y_data)):
            plt.plot(np.convolve(np.log(
                histories[model_name].history[y_data[j]+'_mse']),
                np.ones(smoothing)/smoothing, mode='valid'),
            'k--', alpha = 0.5)
            plt.plot(np.convolve(np.log(
                histories[model_name].history['val_'+y_data[j]+'_mse']),
                np.ones(smoothing)/smoothing, mode='valid'),
            'r--', alpha = 0.5)

            plt.plot(epochs[::smoothing], np.convolve(np.log(
                histories[model_name].history[y_data[j]+'_mse']),
                np.ones(smoothing)/smoothing, mode='valid')[::smoothing],
            'k'+markers[j], label = y_data[j], alpha = 0.5)
            plt.plot(epochs[::smoothing], np.convolve(np.log(
                histories[model_name].history['val_'+y_data[j]+'_mse']),
                np.ones(smoothing)/smoothing, mode='valid')[::smoothing],
            'r'+markers[j], alpha = 0.5)

        if i == 0:
            plt.legend(frameon = False)
        plt.xlabel('epochs')
        plt.ylabel('log MSE')
        plt.title(model_name + ' ({} params)'.format(utils.count_parameters(mod)))
        plt.gca().set_aspect(1./plt.gca().get_data_ratio())

    plt.tight_layout()
