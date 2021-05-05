import os
import unittest
import utils
import pandas as pd
import numpy as np

class MyUtilsTests(unittest.TestCase):
    def test_load_data(self):
        filename = 'test_data.csv'
        data = utils.load_data(filename, filepath=['tests',])
        self.assertEqual(isinstance(data,pd.DataFrame), True)

        filename = 'dummy.csv'
        try:
            data = utils.load_data(filename)
        except Exception as e:
            self.assertEqual(isinstance(e,FileNotFoundError), True)
    
    def test_chroma_train_test_split(self):
        dummy = pd.DataFrame({'x1': [1, 2, 7], 'y1': [2, 1, 8],'x2': [3, 4,9], 'y2': [3, 4, 0], 'cut 1':[0,1,0]})
        trainx, testx, trainy, testy = utils.chroma_train_test_split(
                                        dummy, x=['x1'], y=['y1'], 
                                        keep_cuts = True, test_size=0.50, random_state=np.random.seed(12))

        self.assertEqual(trainx.values, pd.DataFrame({'x1': [7]}).values)
        self.assertEqual(trainy.values, pd.DataFrame({'y1': [8]}).values)

        try:
            trainx, testx, trainy, testy = utils.chroma_train_test_split(
                                dummy, x=['var'], y=['y1'], 
                                keep_cuts = True, test_size=0.50, random_state=np.random.seed(12))
        except Exception as e:
            self.assertEqual(isinstance(e,Exception), True)