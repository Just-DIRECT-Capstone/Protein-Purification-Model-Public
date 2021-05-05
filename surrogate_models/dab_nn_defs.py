import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
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
def create_model_inputs(FEATURE_NAMES):
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float64
        )
    return inputs

# create layer with mean and std for all outputs
def create_model_outputs_prob(TARGET_NAMES,features):
    
    # Create a probabilistic output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.

    outputs = []
    for target_name in TARGET_NAMES:
        distribution_params = layers.Dense(units=2)(features)
        outputs.append(tfp.layers.IndependentNormal(1,
            name=target_name, dtype=tf.float64
        )(distribution_params))
    
    return outputs
def create_model_outputs_det(TARGET_NAMES,features):
    outputs = []
    for target_name in TARGET_NAMES:
        outputs.append(layers.Dense(units=1, name = target_name)(features))
    return outputs

def create_probablistic_bnn_model(FEATURE_NAMES, TARGET_NAMES, train_size, n_outputs, hidden_units, name = 'PBNN'):
    inputs = create_model_inputs(FEATURE_NAMES)
    features = layers.concatenate(list(inputs.values()))

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


def run_experiment(model, loss, learning_rate, num_epochs, train_dataset, test_dataset, verbose = 1):

    logdir="surrogate_models/.logs/"+ model.name +'_'+ datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.MeanSquaredError()]
    )

    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset, callbacks=[tensorboard_callback], verbose = verbose)
    print("Model training finished.")
    rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train MSE: {round(np.sum(rmse), 3)}")

    print("Evaluating model performance...")
    rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test MSE: {round(np.sum(rmse), 3)}")

