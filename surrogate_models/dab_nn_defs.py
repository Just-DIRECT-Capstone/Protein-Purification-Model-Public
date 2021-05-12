import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from datetime import datetime

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# learnable parameters for this distribution are the means, variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


# create layer for all model inputs
def create_model_inputs_old(FEATURE_NAMES):
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,),
        )
    return inputs

def create_model_inputs(FEATURE_NAMES):
    return layers.Input(shape=(len(FEATURE_NAMES)),name='input')

# create layer with mean and std for all outputs
def create_model_outputs_prob(TARGET_NAMES,features):
    
    # Create a probabilistic output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.

    outputs = []
    for target_name in TARGET_NAMES:
        distribution_params = layers.Dense(units=2)(features)
        outputs.append(tfp.layers.IndependentNormal(1,
            name=target_name,
        )(distribution_params))
    
    return outputs

def create_model_outputs_det_old(TARGET_NAMES,features):
    outputs = []
    for target_name in TARGET_NAMES:
        outputs.append(layers.Dense(units=1, name = target_name)(features))
    return outputs

def create_model_outputs_det(TARGET_NAMES,features):
    return layers.Dense(units=len(TARGET_NAMES), name = 'output')(features)

def create_deterministic_nn(FEATURE_NAMES, TARGET_NAMES, hidden_units, name = 'DNN'):
    inputs = create_model_inputs(FEATURE_NAMES)
    features = inputs #layers.concatenate(list(inputs.values()))

    # Create hidden layers using the Dense layer.
    for units in hidden_units:
        features = layers.Dense(
            units=units,
            activation="sigmoid",
        )(features)

    outputs = create_model_outputs_det(TARGET_NAMES,features)

    model = keras.Model(inputs=inputs, outputs=outputs, name = name)
    return model

def create_deterministic_linear_regressor(FEATURE_NAMES, TARGET_NAMES, name = 'DLR'):
    return create_deterministic_nn(FEATURE_NAMES, TARGET_NAMES, hidden_units = [1,], name = name)

def create_probabilistic_linear_regressor(FEATURE_NAMES, TARGET_NAMES, train_size, hidden_units, name = 'PLR'):
    pass

def create_probabilistic_nn(FEATURE_NAMES, TARGET_NAMES, train_size, hidden_units, name = 'PNN'):
    inputs = create_model_inputs(FEATURE_NAMES)
    features = inputs #layers.concatenate(list(inputs.values()))

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            activation="sigmoid",
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight= 1 / train_size,
        )(features)

    outputs = create_model_outputs_det(TARGET_NAMES,features)

    model = keras.Model(inputs=inputs, outputs=outputs, name = name)
    return model


def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


def run_experiment(model, loss, optimizer = keras.optimizers.Adam, learning_rate = 0.1, num_epochs = 1, train_dataset:tuple = None, test_dataset:tuple = None, verbose = 1):

    logdir="surrogate_models/.logs/"+ model.name +'_'+ datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.compile(
        optimizer=optimizer(learning_rate=learning_rate),
        loss=loss,
    )

    print("Start training the model {} ...".format(model.name))
    model.fit(x = train_dataset[0], y = train_dataset[1], epochs=num_epochs, validation_data=test_dataset, callbacks=[tensorboard_callback], verbose = verbose)
    print("Model training finished.")
    rmse = model.evaluate(x = train_dataset[0], y = train_dataset[1], verbose=0)
    print(f"Train MSE: {round(np.sum(rmse), 3)}")

    print("Evaluating model performance...")
    rmse = model.evaluate(x = test_dataset[0], y = test_dataset[1], verbose=0)
    print(f"Test MSE: {round(np.sum(rmse), 3)}")

