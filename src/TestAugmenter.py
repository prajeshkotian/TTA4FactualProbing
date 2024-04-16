import unittest
from unittest.mock import patch, Mock
import pandas as pd
from modules.augmenter import word_swapping, stopword_filtering, back_translation

class TestAugmenter(unittest.TestCase):
    def test_word_swapping(self):
        test_df = pd.DataFrame({
            'prompt': ['What is the capital of France?']
        })
        num_return_sequences = 10
        label = 'test'

        result_df_wordswap = word_swapping(test_df, 'wordswap', num_return_sequences, label)
        result_df_wordnet = word_swapping(test_df, 'wordnet', num_return_sequences, label)

        print('\n')
        print('word_swapping')
        print('\n')
        print('----wordswap----')
        print(result_df_wordswap)
        self.assertIsInstance(result_df_wordswap, pd.DataFrame)
        assert len(result_df_wordswap) == 10
        print('\n')
        print('----wordnet----')
        print(result_df_wordnet)
        self.assertIsInstance(result_df_wordnet, pd.DataFrame)
        assert len(result_df_wordnet) == 10
        print('\n')

    def test_stopword_filtering(self):
        test_df = pd.DataFrame({
            'prompt': ['What is the capital of France?']
        })
        num_return_sequences = 1
        label = 'test'

        result_df = stopword_filtering(test_df, num_return_sequences, label)

        print('\n')
        print('stopword_filtering')
        print(result_df)
        print('\n')
        self.assertIsInstance(result_df, pd.DataFrame)
        assert len(result_df) == 1

    def test_back_translation(self):
        test_df = pd.DataFrame({
            'prompt': ['What is the capital of France?']
        })
        num_return_sequences = 10
        label = 'test'

        result_df = back_translation(test_df, 'fr', num_return_sequences, label)

        print('\n')
        print('back_translation')
        print(result_df)
        print('\n')
        self.assertIsInstance(result_df, pd.DataFrame)
        assert len(result_df) == 10


    

if __name__ == '__main__':
    unittest.main()
