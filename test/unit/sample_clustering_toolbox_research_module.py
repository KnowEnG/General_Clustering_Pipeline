# -*- coding: utf-8 -*-
"""
Created on WEd Aug  17 2016

@author: sobh
@author: The Gene Sets Characterization dev team
"""
import numpy as np

def get_clustered_spreadsheet(n_clusters, cluster_width, row_multiplier=1):
    """ synthetic spreadsheet data with n_clusters (2 : 7) of n_clusters * cluster_width columns """
    n_clusters_array = np.array([3,4])
    nrows = np.product(n_clusters_array*row_multiplier)
    ncols = n_clusters * cluster_width
    SSArr = np.zeros((nrows, ncols))
    row_seg_len = int(nrows / n_clusters)
    row_0 = 0
    col_0 = 0
    row_fin = row_seg_len
    col_fin = cluster_width
    SSArr[row_0:row_fin, col_0:col_fin] += 1
    for k in range(0, n_clusters-1):
        row_0 = row_fin
        col_0 = col_fin
        row_fin = row_0 + row_seg_len
        col_fin = col_0 + cluster_width
        SSArr[row_0:row_fin, col_0:col_fin] += 1

    return SSArr

def get_square_3_cluster_spreadsheet(cluster_rows=3):
    d = cluster_rows
    I = np.ones((d,d))
    oh = np.zeros((d,d))
    A = np.concatenate([I,oh,oh],axis=1)
    B = np.concatenate([oh,I,oh],axis=1)
    C = np.concatenate([oh,oh,I],axis=1)
    spreadsheet = np.concatenate([A,B,C],axis=0)
    return spreadsheet

def get_long_3_cluster_spreadsheet(cluster_rows=3, rows_double=3):
    d = cluster_rows
    I = np.ones((d,d))
    oh = np.zeros((d,d))
    A = np.concatenate([I,oh,oh],axis=1)
    B = np.concatenate([oh,I,oh],axis=1)
    C = np.concatenate([oh,oh,I],axis=1)
    for g in range(0, rows_double):
        A = np.concatenate([A,A],axis=0)
        B = np.concatenate([B,B],axis=0)
        C = np.concatenate([C,C],axis=0)

    spreadsheet = np.concatenate([A,B,C],axis=0)
    return spreadsheet


def get_wide_3_cluster_spreadsheet(cluster_rows=3):
    I = np.ones((cluster_rows))
    oh = np.zeros((cluster_rows))
    spreadsheet = np.zeros((3, cluster_rows * 3))
    spreadsheet[0, :] = np.concatenate([I, oh, oh])
    spreadsheet[1, :] = np.concatenate([oh, I, oh])
    spreadsheet[2, :] = np.concatenate([oh, oh, I])
    return spreadsheet

def get_hclust_sample_data(nrows, ncols, k):
    """ get synthetic data for hclustering

    Args:
        nrows: number of rows
        ncols: number of columns

    Returns:
        X: spreadsheet
        H: clusters
    """
    W = np.random.rand(nrows, k)
    H0 = np.random.rand(k, ncols)

    C = np.argmax(H0, axis=0)
    H = np.zeros(H0.shape)
    for row in range(0, max(C) + 1):
        rowdex = C == row
        H[row, rowdex] = 1

    X = W.dot(H)

    return X, H


def get_cluster_indices_list(a_arr):
    """ get the list of sets of positive integers in the input array where a set
        is the index of where equal values occur for all equal values in the array

    Args:
        a_arr: array of positive integers

    Returns:
        cluster_list: list of lists where each list is the indecies of the members
            of that set, and the lists are ordered by the first member of each.
    """
    idx_arr = np.arange(0, a_arr.size)
    a_arr_unique = np.unique(a_arr)
    tmp_list = []
    for v in a_arr_unique:
        tmp_list.append(idx_arr[a_arr == v])

    len_tmp_list = len(tmp_list)
    first_member_array = np.int_(np.zeros(len_tmp_list))
    for m in range(0, len_tmp_list):
        tmp = tmp_list[m]
        first_member_array[m] = int(tmp[0])

    list_order = np.int_(np.argsort(first_member_array))
    cluster_list = []
    for t in list_order:
        cluster_list.append(tmp_list[t])

    return cluster_list


def get_test_paramters_dictionary():
    test_parameters = {
        'test_directory': '.',
        'method': 'hclustering',
        'method_1': 'kmeans`f',
        'spreadsheet_name_full_path': '../../data/spreadsheets/tcga_ucec_somatic_mutation_data.df',
        'results_directory': './tmp',
        'number_of_clusters': 3,
        'processing_method': 'serial',
        'processing_method_1': 'serial',
        'processing_method_2': 'parallel',
        'top_number_of_rows': 10
        }

    return test_parameters
