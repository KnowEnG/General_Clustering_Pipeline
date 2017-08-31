import unittest
from unittest import TestCase
import general_clustering_toolbox as tl
import numpy as np

class TestRun_ClustLink(TestCase):
    def setUp(self):
        self.run_parameters = {"spreadsheet_name_full_path":"../../data/spreadsheets/zTEST_1_gene_sample.tsv",
                               "results_directory":"./tmp",
                               "method": "hclust_link",
                               "top_number_of_rows": 2,
                               "nearest_neighbors": 3,
                               "number_of_clusters": 2}

    def tearDown(self):
        del self.run_parameters

    def test_run_hclust_link(self):
        ret = tl.run_hclust_link(self.run_parameters)
        expected = np.array([1, 0, 0, 0])
        self.assertEqual(np.array_equal(ret, expected), True)

if __name__ == '__main__':
    unittest.main()