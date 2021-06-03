"""imports"""
import os
import unittest
import utils
import pandas as pd

class MyUtilsTests(unittest.TestCase):
    """unit tests"""
    def test_load_data(self):
        """tests data loading"""
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
        """tests data splitting"""
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
