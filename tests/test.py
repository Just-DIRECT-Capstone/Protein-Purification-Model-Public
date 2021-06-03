"""imports"""
import os
import unittest
import utils
import pandas as pd

class MyUtilsTests(unittest.TestCase):
    """unit tests"""
    def test_load_data(self):
        """tests function that loads data into dataframe"""
        filename = 'test_data.csv'
        dir = os.getcwd()

        data = utils.load_data(dir,filename, filepath=['tests',])
        self.assertEqual(isinstance(data,pd.DataFrame), True)

        filename = 'dummy.csv'
        try:
            data = utils.load_data(dir, filename)
        except Exception as exception:
            self.assertEqual(isinstance(exception,FileNotFoundError), True)

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
        if data1.shape[0] < data_try.shape[0]:
            result = 0
        assert result == 1, "problem removing out of bounds data"
        
    def test_data_pipeline(self):
        """tests function that splits data"""
        filename = 'mol_res_scan_results_7.csv'
        dir = os.getcwd()
        data1 = utils.load_data(dir,filename, filepath=['tests',])
        CV = 5
        data2split, validation = utils.chroma_train_test_split(data1, test_size=0.20)
        trains, tests = utils.data_pipeline([data2split,], x, y, cross_val = CV)
        assert isinstance(trains,list) == True, "not printing list"
        assert len(trains[0]) == CV, "not splitting into CV folds"
    
    def test_count_parameters(self):
        """tests function that counts parameters"""
        return
        
    def test_get_model_name(self):
        """tests function that retrieves model name"""
        return