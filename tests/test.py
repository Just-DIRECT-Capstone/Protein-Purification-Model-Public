"""imports"""
import os
import unittest
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

import utils
import surrogate_models.nn_defs as engine

class MyUtilsTests(unittest.TestCase):
    """unit tests for utility functions"""

    def test_load_data(self):
        """tests function that loads data into dataframe"""
        filename = 'test_data.csv'
        dir = os.getcwd()

        data = utils.load_data(dir,filename, filepath=['tests',])
        self.assertTrue(isinstance(data,pd.DataFrame))

        filename = 'dummy.csv'
        try:
            data = utils.load_data(dir, filename)
        except Exception as exception:
            self.assertTrue(isinstance(exception,FileNotFoundError))

    def test_chroma_train_test_split(self):
        """tests function that splits data into testing and training"""
        dummy = pd.DataFrame({'x1': [1, 2, 7], 'y1': [2, 1, 8],'x2': [3, 4,9],
        'y2': [3, 4, 0], 'cut 1':[0,1,0]})
        trainx, testx, trainy, testy = utils.chroma_train_test_split(
                                        dummy, x_data=['x1'], y_data=['y1'],
                                        keep_cuts = True, test_size=0.50, random_seed=12)

        self.assertEqual(trainx.values, pd.DataFrame({'x1': [7]}).values)
        self.assertEqual(trainy.values, pd.DataFrame({'y1': [8]}).values)

        try:
            trainx, testx, trainy, testy = utils.chroma_train_test_split(
                                dummy, x_data=['var'], y_data=['y1'],
                                keep_cuts = True, test_size=0.50, random_seed=12)
        except Exception as exception:
            self.assertEqual(isinstance(exception,Exception), True)
    
    def test_preprocessing(self):
        """tests function that removes physically impossible data"""
        filename = 'mol_res_scan_results_7.csv'
        dir = os.getcwd()
        data1 = utils.load_data(dir,filename, filepath=['tests',])
        data_try = utils.preprocessing([data1,], bounds = {'yield':[0,1],'purity':[0,1]})[0]
        if data1.shape[0] > data_try.shape[0]:
            result = 1
        assert result == 1, "problem removing out of bounds data"
        
    def test_data_pipeline(self):
        """tests function that splits data"""
        filename = 'mol_res_scan_results_7.csv'
        dir = os.getcwd()
        data1 = utils.load_data(dir,filename, filepath=['tests',])
        CV = 5
        x = [*data1.columns[:2],*data1.columns[4:]]
        y = data1.columns[2:4]
        data2split, validation = utils.chroma_train_test_split(data1, test_size=0.20)
        trains, tests = utils.data_pipeline([data2split,], x_data=x, y_data=y, cross_val = CV)
        assert isinstance(trains,list) == True, "not printing list"
        assert len(trains[0]) == CV, "not splitting into CV folds"
    
    def test_count_parameters(self):
        """tests function that counts parameters"""
        data = utils.load_data(os.getcwd(),'mol_res_scan_results_7.csv', filepath=['tests',])
        dummy = engine.create_deterministic_nn(
            feature_names = [*data.columns[:2],*data.columns[4:]],
            target_names = data.columns[2:4],
            hidden_units = [16,8,4],
            name = 'dummy'
            )
        n_params = utils.count_parameters(dummy)
        assert isinstance(n_params, int), "not returning an integer"
        assert n_params == 486, "incorrect number of parameters"

    def test_load_model(self):
        """tests function that loads pretrained models"""
        dummy_model, dummy_settings = utils.load_model(os.path.join(os.getcwd(),
            'surrogate_models','saved_models','DNN_mol_res_scan_results_7'))

        assert isinstance(dummy_settings, dict), "not returning a settings dictionary"
        assert isinstance(dummy_model, tf.keras.Model), "not returning a tensorflow model"

class MyNNTests(unittest.TestCase):
    """unit tests for model functions"""

    def test_prior(self):
        """tests function that defines priors"""

        kernel_size = 3
        bias_size = 3
        dummy_prior = engine.prior(kernel_size, bias_size)
        assert isinstance(dummy_prior, tf.keras.Model), "not returning a tensorflow model"

    def test_posterior(self):
        """tests function that defines posteriors"""

        kernel_size = 3
        bias_size = 3
        dummy_post = engine.posterior(kernel_size, bias_size)
        assert isinstance(dummy_post, tf.keras.Model), "not returning a tensorflow model"

    def test_create_model_outputs_det(self):
        """tests function that defines determinsitc outputs"""

        features = ['a','b']
        targets = ['c']
        inputs = engine.create_model_inputs(features)
        dummy_outputs = engine.create_model_outputs_det(targets,inputs)
        assert isinstance(dummy_outputs, list), "not returning a list"
        assert tf.keras.backend.is_keras_tensor(dummy_outputs[0]), "not a tensor"

    def test_create_model_outputs_prob(self):
        """tests function that defines probabilistic outputs"""

        features = ['a','b']
        targets = ['c']
        inputs = engine.create_model_inputs(features)
        dummy_outputs = engine.create_model_outputs_prob(targets,inputs)
        assert isinstance(dummy_outputs, list), "not returning a list"
        assert tf.keras.backend.is_keras_tensor(dummy_outputs[0]), "not a tensor"

    def test_create_deterministic_nn(self):
        """tests function that defines NNs"""
        dummy = engine.create_deterministic_nn(
            feature_names = ['a','b'],
            target_names = ['c'],
            hidden_units = [16,8,4],
            name = 'dummy'
            )
        assert isinstance(dummy, tf.keras.Model), "not returning a tensorflow model"
        assert isinstance(dummy.layers[-1],tf.keras.layers.Dense), "incorrect output layer"

    def test_create_probabilistic_nn(self):
        """tests function that defines NNs"""
        dummy = engine.create_deterministic_nn(
            feature_names = ['a','b'],
            target_names = ['c'],
            hidden_units = [16,8,4],
            name = 'dummy',
            )
        assert isinstance(dummy, tf.keras.Model), "not returning a tensorflow model"
        assert isinstance(dummy.layers[-1],tfp.layers.IndependentNormal), "incorrect output layer"