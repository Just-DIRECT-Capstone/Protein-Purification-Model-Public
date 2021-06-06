"""imports"""
from copy import deepcopy
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

def prior(kernel_size, bias_size): #removed dtype=None, unused argument
    """define the prior weight distribution as Normal of mean=0 and stddev=1"""
    number = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(number), scale_diag=tf.ones(number)
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    """define variational posterior weight distribution as multivariate Gaussian
    learnable parameters are means, variances, and covariances"""
    number = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(number), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(number),
        ]
    )
    return posterior_model

def create_model_inputs(feature_names):
    """create layer for all model inputs"""
    return layers.Input(shape=(len(feature_names)),name='input')

def create_model_outputs_prob(target_names,features):
    """create layer with mean and std for all outputs"""
    # Create a probabilistic output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.

    outputs = []
    for target_name in target_names:
        distribution_params = layers.Dense(units=2, name = target_name+'_params')(features)
        outputs.append(tfp.layers.IndependentNormal(1,name=target_name,)(distribution_params))
    return outputs

def create_model_outputs_det(target_names,features):
    """create model outputs"""
    outputs = []
    for target_name in target_names:
        outputs.append(layers.Dense(units=1, name = target_name)(features))
    return outputs

def create_deterministic_nn(feature_names, target_names, hidden_units, name = 'DNN', out = 'D'):
    """function for deterministic neural network based on Nagrath"""
    inputs = create_model_inputs(feature_names)
    features = inputs #layers.concatenate(list(inputs.values()))

    # Create hidden layers using the Dense layer.
    for units in hidden_units:
        features = layers.Dense(
            units=units,
            activation="sigmoid",
        )(features)

    if out == 'D':
        outputs = create_model_outputs_det(target_names,features)
    if out == 'P':
        outputs = create_model_outputs_prob(target_names,features)

    model = keras.Model(inputs=inputs, outputs=outputs, name = name)
    return model

def create_probabilistic_nn(feature_names, target_names, hidden_units, name = 'PNN'):
    """create probabilistic neural network"""
    return create_deterministic_nn(feature_names, target_names, hidden_units,
            name = name, out = 'P')

def create_deterministic_linear_regressor(feature_names, target_names, name = 'DLR'):
    """function to create deterministic linear regression model"""
    return create_deterministic_nn(feature_names, target_names, hidden_units = [1,],
            name = name, out = 'D')

def create_probabilistic_linear_regressor(feature_names, target_names, name = 'PLR'):
    """function to create probabilistic linear regression model"""
    return create_deterministic_nn(feature_names, target_names, hidden_units = [1,],
            name = name, out = 'P')

# def create_probabilistic_nn_old(feature_names, target_names, train_size,
#         hidden_units, name = 'PNN'):
#     """old nn??"""
#     inputs = create_model_inputs(feature_names)
#     features = inputs #layers.concatenate(list(inputs.values()))

#     # Create hidden layers with weight uncertainty using the DenseVariational layer.
#     for units in hidden_units:
#         features = tfp.layers.DenseVariational(
#             units=units,
#             activation="sigmoid",
#             make_prior_fn=prior,
#             make_posterior_fn=posterior,
#             kl_weight= 1 / train_size,
#         )(features)

#     outputs = create_model_outputs_det(target_names,features)

#     model = keras.Model(inputs=inputs, outputs=outputs, name = name)
#     return model

def negative_loglikelihood(targets, estimated_distribution):
    """function that returns negative_loglikelihood"""
    return -estimated_distribution.log_prob(targets)

def run_experiment(model, loss, loss_weights, optimizer = keras.optimizers.Adam,
        learning_rate = 0.1, num_epochs = 1, train_dataset:list = None,
        test_dataset:list = None, verbose = 1, log = False):
    """running experiments with each model"""
    train_data = deepcopy(train_dataset)
    test_data = deepcopy(test_dataset)

    # should check if data is in dataframes
    try:
        train_data[0] = train_data[0].to_numpy()
        test_data[0] = test_data[0].to_numpy()
        train_data[1] = {k:np.array(v) for k, v in train_data[1].to_dict('list').items()}
        test_data[1] = {k:np.array(v) for k, v in test_data[1].to_dict('list').items()}
    except:
        pass

    if log:
        logdir="surrogate_models/.logs/"+ model.name +'_'+ datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.compile(
        optimizer = optimizer(learning_rate=learning_rate),
        loss = loss,
        loss_weights = loss_weights,
        metrics = ['mse'],
    )

    print("Start training the model {} ...".format(model.name))

    history = model.fit(x = train_data[0], y = train_data[1], epochs=num_epochs,
            validation_data=tuple(test_data), callbacks=[tensorboard_callback],
            verbose = verbose) if log else model.fit(x = train_data[0], y = train_data[1],
                    epochs=num_epochs, validation_data=tuple(test_data), verbose = verbose)
    print("Evaluating model performance...")

    train_hat = model(train_data[0])
    test_hat = model(test_data[0])

    rmse = multi_mse(train_dataset[1], train_hat)
    print(f"Train MSE: {round(np.sum(rmse)*100, 3)}")

    rmse = multi_mse(test_dataset[1], test_hat)
    print(f"Test MSE: {round(np.sum(rmse)*100, 3)}")
    return history

def multi_mse(true, pred):
    """function to calculate the MSE"""
    try:
        pred = [t.mean() for t in pred]
    except:
        pass

    pred = np.array([p.numpy() for p in pred]).squeeze()
    return tf.keras.metrics.mean_squared_error(true.to_numpy().T, pred)
