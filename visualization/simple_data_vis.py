import numpy as np
import matplotlib.pyplot as plt
import utils
from surrogate_models.dab_nn_defs import multi_mse

def histograms(data,x,y):
    n_outputs = len(y)
    n_inputs = len(x)
    n = int(np.ceil(n_inputs**0.5))

    fig, ax = plt.subplots(n, n, figsize = (12,12))
    fig.tight_layout()

    for a,var in zip(ax.flat,x):
        data[var].hist(ax = a)
        a.set_title(var)

    fig, ax = plt.subplots(n_outputs, 1, figsize = (3,6))
    fig.tight_layout()

    for a,var in zip(ax.flat,y):
        data[var].hist(ax = a)
        a.set_title(var)

    return fig, ax

def scatter_hats(models, train, test, settings, N = 50):
    y = train[1].columns
    markers = ['.','^'] # ok for two outputs only

    sample_inputs_train = train[0].sample(N)
    sample_outputs_train = train[1].loc[sample_inputs_train.index]
    sample_inputs_test = test[0].sample(N)
    sample_outputs_test = test[1].loc[sample_inputs_test.index]

    plt.figure(figsize = (10*len(models)/3,10*len(models)))
    for i,m in enumerate(models):
        model_name = utils.get_model_name(m, settings['dataset'])
        plt.subplot(1,len(models),i+1)

        train_hat = m(sample_inputs_train.to_numpy())
        test_hat = m(sample_inputs_test.to_numpy())

        for j in range(len(y)):
            try:
                plt.errorbar(sample_outputs_train.to_numpy()[:,j], train_hat[j].numpy(),
                            marker = markers[j], color = 'k', alpha = 0.5, ls = 'none',  label = y[j])
                plt.errorbar(sample_outputs_test.to_numpy()[:,j], test_hat[j].numpy(),
                            marker = markers[j], color = 'r', alpha = 0.5, ls = 'none')
            except:
                plt.errorbar(sample_outputs_train.to_numpy()[:,j], train_hat[j].mean().numpy(), yerr = train_hat[j].stddev().numpy().squeeze(),
                            marker = markers[j], color = 'k', alpha = 0.5, ls = 'none',  label = y[j])
                plt.errorbar(sample_outputs_test.to_numpy()[:,j], test_hat[j].mean().numpy(), yerr = test_hat[j].stddev().numpy().squeeze(),
                            marker = markers[j], color = 'r', alpha = 0.5, ls = 'none')

        plt.title(model_name + ' ({} params)'.format(utils.count_parameters(m)))
        xmax = 1.1
        plt.gca().set_aspect('equal')
        plt.xlim([0,xmax])
        plt.ylim([0,xmax])
        plt.plot([0,xmax],[0,xmax],'k',alpha=0.25)
        plt.xlabel('true')
        plt.ylabel('predicted')
        if i == 0: plt.legend(frameon = False)

        rmse = multi_mse(train[1], m(train[0].to_numpy()))
        plt.text(0.5,0.12,f"Train MSE: {round(np.sum(rmse)*100, 2)}", color = 'k')
        rmse = multi_mse(test[1], m(test[0].to_numpy()))
        plt.text(0.5,0.04,f"Test MSE: {round(np.sum(rmse)*100, 2)}", color = 'r')

    plt.tight_layout()
    plt.text(2, .80, '.', fontsize=1)
    # could be a loop
    plt.text(1.20, .90, 'Dataset: '+settings['dataset'][:-4], fontsize=12)
    plt.text(1.20, .70, 'Optimizer: '+settings['optimizer'], fontsize=12)
    plt.text(1.20, .50, 'Learning Rate: '+str(settings['learning_rate']), fontsize=12)
    plt.text(1.20, .30, 'Loss Weights: '+str(settings['loss_weights']), fontsize=12)
    plt.text(1.20, .10, 'Epochs: '+str(settings['epochs']), fontsize=12)

    return

def training_curves(models, y, settings, histories, smoothing = 1):
    epochs = np.arange(settings['epochs'])
    markers = ['.','^'] # ok for two outputs only
    plt.figure(figsize = (10*len(models)/3,10*len(models)))
    for i,m in enumerate(models):
        model_name = utils.get_model_name(m, settings['dataset'])
        plt.subplot(1,len(models),i+1)
        for j in range(len(y)):
            plt.plot(np.convolve(np.log(
                histories[model_name].history[y[j]+'_mse']), np.ones(smoothing)/smoothing, mode='valid'),
            'k--', alpha = 0.5)
            plt.plot(np.convolve(np.log(
                histories[model_name].history['val_'+y[j]+'_mse']), np.ones(smoothing)/smoothing, mode='valid'),
            'r--', alpha = 0.5)

            plt.plot(epochs[::smoothing], np.convolve(np.log(
                histories[model_name].history[y[j]+'_mse']), np.ones(smoothing)/smoothing, mode='valid')[::smoothing],
            'k'+markers[j], label = y[j], alpha = 0.5)
            plt.plot(epochs[::smoothing], np.convolve(np.log(
                histories[model_name].history['val_'+y[j]+'_mse']), np.ones(smoothing)/smoothing, mode='valid')[::smoothing],
            'r'+markers[j], alpha = 0.5)

        if i == 0: plt.legend(frameon = False)
        plt.xlabel('epochs')
        plt.ylabel('log MSE')
        plt.title(model_name + ' ({} params)'.format(utils.count_parameters(m)))
        plt.gca().set_aspect(1./plt.gca().get_data_ratio())

    plt.tight_layout()
    return
