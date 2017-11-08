"""
@author: The KnowEnG dev team
"""
import os
import numpy as np
import pandas as pd

from sklearn.cluster   import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics   import silhouette_score

from sklearn.metrics.pairwise import pairwise_distances

from scipy.sparse import csr_matrix

import knpackage.toolbox as kn
import knpackage.distributed_computing_utils as dstutil

import general_clustering_eval_toolbox as cluster_eval

def run_cc_link_hclust(run_parameters):
    """ wrapper: call sequence to perform hclust with
        consensus clustering and write results.

    Args:
        run_parameters: parameter set dictionary.
    """
    tmp_dir = 'tmp_cc_link_hclust'
    run_parameters = update_tmp_directory(run_parameters, tmp_dir)

    processing_method          = run_parameters['processing_method']
    number_of_bootstraps       = run_parameters['number_of_bootstraps']
    number_of_clusters         = run_parameters['number_of_clusters']
    nearest_neighbors          = run_parameters['nearest_neighbors']
    affinity_metric            = run_parameters['affinity_metric']
    linkage_criterion          = run_parameters['linkage_criterion']
    spreadsheet_name_full_path = run_parameters['spreadsheet_name_full_path']

    spreadsheet_df             = kn.get_spreadsheet_df(spreadsheet_name_full_path)
    spreadsheet_mat            = spreadsheet_df.as_matrix()
    number_of_samples          = spreadsheet_mat.shape[1]

    if processing_method == 'serial':
        for sample in range(0, number_of_bootstraps):
            run_cc_link_hclust_clusters_worker(spreadsheet_mat, run_parameters, sample)

    elif processing_method == 'parallel':
        find_and_save_cc_link_hclust_clusters_parallel(spreadsheet_mat, run_parameters, number_of_bootstraps)

    elif processing_method == 'distribute':
        func_args = [spreadsheet_mat, run_parameters]
        dependency_list = [run_cc_link_hclust_clusters_worker, kn.save_a_clustering_to_tmp, dstutil.determine_parallelism_locally]
        dstutil.execute_distribute_computing_job(run_parameters['cluster_ip_address'],
                                                 number_of_bootstraps,
                                                 func_args,
                                                 find_and_save_cc_link_hclust_clusters_parallel,
                                                 dependency_list)
    else:
        raise ValueError('processing_method contains bad value.')

    consensus_matrix = kn.form_consensus_matrix(run_parameters, number_of_samples)
    labels           = perform_link_hclust( consensus_matrix
                                          , number_of_clusters
                                          , nearest_neighbors 
                                          , affinity_metric 
                                          , linkage_criterion) 

    sample_names     = spreadsheet_df.columns

    save_consensus_clustering            (consensus_matrix, sample_names, labels, run_parameters)
    save_final_samples_clustering        (sample_names, labels, run_parameters)
    save_spreadsheet_and_variance_heatmap(spreadsheet_df, labels, run_parameters)

    kn.remove_dir(run_parameters["tmp_directory"])


def find_and_save_cc_link_hclust_clusters_parallel(spreadsheet_mat, run_parameters, local_parallelism):
    """ central loop: compute components for the consensus matrix by hclust.

    Args:
        spreadsheet_mat: genes x samples matrix.
        run_parameters: dictionary of run-time parameters.
        number_of_cpus: number of processes to be running in parallel
    """
    import knpackage.distributed_computing_utils as dstutil

    jobs_id         = range(0, local_parallelism)
    zipped_arguments= dstutil.zip_parameters(spreadsheet_mat, run_parameters, jobs_id)

    if 'parallelism' in run_parameters:
        parallelism = dstutil.determine_parallelism_locally(local_parallelism, run_parameters['parallelism'])
    else:
        parallelism = dstutil.determine_parallelism_locally(local_parallelism)

    dstutil.parallelize_processes_locally(run_cc_link_hclust_clusters_worker, zipped_arguments, parallelism)


def run_cc_link_hclust_clusters_worker(spreadsheet_mat, run_parameters, sample):
    """Worker to execute hclust in a single process

    Args:
        spreadsheet_mat: genes x samples matrix.
        run_parameters: dictionary of run-time parameters.
        sample: each loops.

    Returns:
        None

    """
    import knpackage.toolbox as kn
    import numpy as np

    np.random.seed(sample)
    rows_sampling_fraction = run_parameters["rows_sampling_fraction"]
    cols_sampling_fraction = run_parameters["cols_sampling_fraction"]
    number_of_clusters     = run_parameters["number_of_clusters"]
    nearest_neighbors      = run_parameters["nearest_neighbors"]
    affinity_metric        = run_parameters['affinity_metric']
    linkage_criterion      = run_parameters['linkage_criterion']

    spreadsheet_mat, sample_permutation = kn.sample_a_matrix(spreadsheet_mat,
                                                             rows_sampling_fraction, cols_sampling_fraction)

    labels                 = perform_link_hclust( spreadsheet_mat.T
                                                , number_of_clusters
                                                , nearest_neighbors 
                                                , affinity_metric
                                                , linkage_criterion )

    h_mat                  = labels_to_hmat(labels, number_of_clusters)
    kn.save_a_clustering_to_tmp(h_mat, sample_permutation, run_parameters, sample)


def perform_link_hclust(spreadsheet_mat, number_of_clusters, nearest_neighbors, affinity_metric, linkage_criterion):
    """ wrapper: call sequence to perform hclust clustering 

    Args:
        spreadsheet_mat: matrix to be clusters by rows
        number_of_clusters: number of clusters requested
    """

    connectivity = kneighbors_graph(spreadsheet_mat, n_neighbors=nearest_neighbors, include_self=False)
    if affinity_metric == 'jaccard' :

        distance_mat = 1.0 - pairwise_distances(spreadsheet_mat,metric='jaccard')
        if linkage_criterion == "ward" : affinity_metric = "euclidean"
        l_method = AgglomerativeClustering( n_clusters   = number_of_clusters
                                          , affinity     = affinity_metric
                                          , connectivity = connectivity
                                          , linkage      = linkage_criterion  ).fit(distance_mat)
    else:
        l_method = AgglomerativeClustering( n_clusters   = number_of_clusters
                                          , affinity     = affinity_metric
                                          , connectivity = connectivity
                                          , linkage      = linkage_criterion  ).fit(spreadsheet_mat)
    labels     = l_method.labels_
 
    return labels

def run_cc_hclust(run_parameters):
    """ wrapper: call sequence to perform hclust with
        consensus clustering and write results.

    Args:
        run_parameters: parameter set dictionary.
    """
    tmp_dir = 'tmp_cc_nmf'
    run_parameters = update_tmp_directory(run_parameters, tmp_dir)

    processing_method          = run_parameters['processing_method']
    affinity_metric            = run_parameters['affinity_metric']
    linkage_criterion          = run_parameters['linkage_criterion']
   
    number_of_bootstraps       = run_parameters['number_of_bootstraps']
    number_of_clusters         = run_parameters['number_of_clusters']
    spreadsheet_name_full_path = run_parameters['spreadsheet_name_full_path']

    spreadsheet_df             = kn.get_spreadsheet_df(spreadsheet_name_full_path)
    spreadsheet_mat            = spreadsheet_df.as_matrix()
    number_of_samples          = spreadsheet_mat.shape[1]

    if processing_method == 'serial':
        for sample in range(0, number_of_bootstraps):
            run_cc_hclust_clusters_worker(spreadsheet_mat, run_parameters, sample)

    elif processing_method == 'parallel':
        find_and_save_cc_hclust_clusters_parallel(spreadsheet_mat, run_parameters, number_of_bootstraps)

    elif processing_method == 'distribute':
        func_args = [spreadsheet_mat, run_parameters]
        dependency_list = [run_cc_hclust_clusters_worker, kn.save_a_clustering_to_tmp, dstutil.determine_parallelism_locally]
        dstutil.execute_distribute_computing_job(run_parameters['cluster_ip_address'],
                                                 number_of_bootstraps,
                                                 func_args,
                                                 find_and_save_cc_hclust_clusters_parallel,
                                                 dependency_list)
    else:
        raise ValueError('processing_method contains bad value.')

    consensus_matrix = kn.form_consensus_matrix(run_parameters, number_of_samples)
    labels           = perform_hclust(consensus_matrix, number_of_clusters, affinity_metric, linkage_criterion)
    sample_names     = spreadsheet_df.columns

    save_consensus_clustering(consensus_matrix, sample_names, labels, run_parameters)
    save_final_samples_clustering(sample_names, labels, run_parameters)
    save_spreadsheet_and_variance_heatmap(spreadsheet_df, labels, run_parameters)

    kn.remove_dir(run_parameters["tmp_directory"])


def find_and_save_cc_hclust_clusters_parallel(spreadsheet_mat, run_parameters, local_parallelism):
    """ central loop: compute components for the consensus matrix by hclust.

    Args:
        spreadsheet_mat: genes x samples matrix.
        run_parameters: dictionary of run-time parameters.
        number_of_cpus: number of processes to be running in parallel
    """
    import knpackage.distributed_computing_utils as dstutil

    jobs_id = range(0, local_parallelism)
    zipped_arguments = dstutil.zip_parameters(spreadsheet_mat, run_parameters, jobs_id)
    if 'parallelism' in run_parameters:
        parallelism = dstutil.determine_parallelism_locally(local_parallelism, run_parameters['parallelism'])
    else:
        parallelism = dstutil.determine_parallelism_locally(local_parallelism)
    dstutil.parallelize_processes_locally(run_cc_hclust_clusters_worker, zipped_arguments, parallelism)


def run_cc_hclust_clusters_worker(spreadsheet_mat, run_parameters, sample):
    """Worker to execute hclust in a single process

    Args:
        spreadsheet_mat: genes x samples matrix.
        run_parameters: dictionary of run-time parameters.
        sample: each loops.

    Returns:
        None

    """
    import knpackage.toolbox as kn
    import numpy as np

    np.random.seed(sample)

    rows_sampling_fraction = run_parameters["rows_sampling_fraction"]
    cols_sampling_fraction = run_parameters["cols_sampling_fraction"]
    number_of_clusters     = run_parameters["number_of_clusters"]
    affinity_metric        = run_parameters['affinity_metric']
    linkage_criterion      = run_parameters['linkage_criterion']

    spreadsheet_mat, sample_permutation = kn.sample_a_matrix(spreadsheet_mat,
                                                             rows_sampling_fraction, cols_sampling_fraction)

    labels                 = perform_hclust(spreadsheet_mat.T, number_of_clusters, affinity_metric, linkage_criterion)
    h_mat                  = labels_to_hmat(labels, number_of_clusters)
    kn.save_a_clustering_to_tmp(h_mat, sample_permutation, run_parameters, sample)



def run_cc_kmeans(run_parameters):
    """ wrapper: call sequence to perform kmeans with
        consensus clustering and write results.

    Args:
        run_parameters: parameter set dictionary.
    """
    tmp_dir = 'tmp_cc_nmf'
    run_parameters             = update_tmp_directory(run_parameters, tmp_dir)

    processing_method          = run_parameters['processing_method']

    number_of_bootstraps       = run_parameters['number_of_bootstraps']
    number_of_clusters         = run_parameters['number_of_clusters']
    spreadsheet_name_full_path = run_parameters['spreadsheet_name_full_path']

    spreadsheet_df             = kn.get_spreadsheet_df(spreadsheet_name_full_path)
    spreadsheet_mat            = spreadsheet_df.as_matrix()
    number_of_samples          = spreadsheet_mat.shape[1]

    if processing_method == 'serial':
        for sample in range(0, number_of_bootstraps):
            run_cc_kmeans_clusters_worker(spreadsheet_mat, run_parameters, sample)

    elif processing_method == 'parallel':
        find_and_save_cc_kmeans_clusters_parallel(spreadsheet_mat, run_parameters, number_of_bootstraps)

    elif processing_method == 'distribute':
        func_args = [spreadsheet_mat, run_parameters]
        dependency_list = [run_cc_kmeans_clusters_worker, kn.save_a_clustering_to_tmp, dstutil.determine_parallelism_locally]
        dstutil.execute_distribute_computing_job(run_parameters['cluster_ip_address'],
                                                 number_of_bootstraps,
                                                 func_args,
                                                 find_and_save_cc_kmeans_clusters_parallel,
                                                 dependency_list)
    else:
        raise ValueError('processing_method contains bad value.')

    consensus_matrix = kn.form_consensus_matrix(run_parameters, number_of_samples)
    labels = kn.perform_kmeans(consensus_matrix, number_of_clusters)

    sample_names = spreadsheet_df.columns
    save_consensus_clustering(consensus_matrix, sample_names, labels, run_parameters)
    save_final_samples_clustering(sample_names, labels, run_parameters)
    save_spreadsheet_and_variance_heatmap(spreadsheet_df, labels, run_parameters)

    kn.remove_dir(run_parameters["tmp_directory"])


def find_and_save_cc_kmeans_clusters_parallel(spreadsheet_mat, run_parameters, local_parallelism):
    """ central loop: compute components for the consensus matrix by kmeans.

    Args:
        spreadsheet_mat: genes x samples matrix.
        run_parameters: dictionary of run-time parameters.
        number_of_cpus: number of processes to be running in parallel
    """
    import knpackage.distributed_computing_utils as dstutil

    jobs_id         = range(0, local_parallelism)
    zipped_arguments= dstutil.zip_parameters(spreadsheet_mat, run_parameters, jobs_id)
    if 'parallelism' in run_parameters:
        parallelism = dstutil.determine_parallelism_locally(local_parallelism, run_parameters['parallelism'])
    else:
        parallelism = dstutil.determine_parallelism_locally(local_parallelism)
    dstutil.parallelize_processes_locally(run_cc_kmeans_clusters_worker, zipped_arguments, parallelism)


def run_cc_kmeans_clusters_worker(spreadsheet_mat, run_parameters, sample):
    """Worker to execute kmeans in a single process

    Args:
        spreadsheet_mat: genes x samples matrix.
        run_parameters: dictionary of run-time parameters.
        sample: each loops.

    Returns:
        None

    """
    import knpackage.toolbox as kn
    import numpy as np

    np.random.seed(sample)
    rows_sampling_fraction = run_parameters["rows_sampling_fraction"]
    cols_sampling_fraction = run_parameters["cols_sampling_fraction"]
    number_of_clusters     = run_parameters["number_of_clusters"]
    spreadsheet_mat, sample_permutation = kn.sample_a_matrix(spreadsheet_mat,
                                                             rows_sampling_fraction, cols_sampling_fraction)

    labels                 = kn.perform_kmeans(spreadsheet_mat.T, number_of_clusters)
    h_mat                  = labels_to_hmat(labels, number_of_clusters)
    kn.save_a_clustering_to_tmp(h_mat, sample_permutation, run_parameters, sample)


def perform_hclust(spreadsheet_mat, number_of_clusters, affinity_metric, linkage_criterion):
    """ wrapper: call sequence to perform hclust clustering 

    Args:
        spreadsheet_mat: matrix to be clusters by rows
        number_of_clusters: number of clusters requested
    """

    if affinity_metric == 'jaccard' :

        distance_mat = 1.0 - pairwise_distances(spreadsheet_mat,metric='jaccard')
        if linkage_criterion == "ward" : affinity_metric = "euclidean"
        l_method = AgglomerativeClustering( n_clusters   = number_of_clusters
                                          , affinity     = affinity_metric
                                          , linkage      = linkage_criterion  ).fit(distance_mat)
    else:
        l_method = AgglomerativeClustering( n_clusters   = number_of_clusters  
                                          , affinity     = affinity_metric  
                                          , linkage      = linkage_criterion  ).fit(spreadsheet_mat)

    labels = l_method.labels_

    return labels


def run_kmeans(run_parameters):
    """ wrapper: call sequence to perform kmeans clustering and save the results.

    Args:
        run_parameters: parameter set dictionary.
    """

    number_of_clusters         = run_parameters['number_of_clusters'        ]

    spreadsheet_name_full_path = run_parameters['spreadsheet_name_full_path']

    spreadsheet_df             = kn.get_spreadsheet_df(spreadsheet_name_full_path)
    spreadsheet_mat            = spreadsheet_df.as_matrix()
    number_of_samples          = spreadsheet_mat.shape[1]

    labels                     = kn.perform_kmeans(spreadsheet_mat.T, number_of_clusters)
    sample_names               = spreadsheet_df.columns

    save_final_samples_clustering        (sample_names  , labels, run_parameters)
    save_spreadsheet_and_variance_heatmap(spreadsheet_df, labels, run_parameters)

    return labels

def run_hclust(run_parameters):
    """ wrapper: call sequence to perform hierchical clustering and save the results.

    Args:
        run_parameters: parameter set dictionary.
    """

    np.random.seed()
    number_of_clusters         = run_parameters['number_of_clusters'        ]
    affinity_metric            = run_parameters['affinity_metric']
    linkage_criterion          = run_parameters['linkage_criterion']

    spreadsheet_name_full_path = run_parameters['spreadsheet_name_full_path']

    spreadsheet_df             = kn.get_spreadsheet_df(spreadsheet_name_full_path)
    spreadsheet_mat            = spreadsheet_df.as_matrix()
    number_of_samples          = spreadsheet_mat.shape[1]

    labels                     = perform_hclust(spreadsheet_mat.T, number_of_clusters, affinity_metric, linkage_criterion)
    sample_names               = spreadsheet_df.columns

    save_final_samples_clustering        (sample_names  , labels, run_parameters)
    save_spreadsheet_and_variance_heatmap(spreadsheet_df, labels, run_parameters)

    return labels

def run_link_hclust(run_parameters):
    """ wrapper: call sequence to perform hierchical clustering using linkage and save the results.

    Args:
        run_parameters: parameter set dictionary.
    """

    np.random.seed()
    nearest_neighbors          = run_parameters['nearest_neighbors'         ]
    number_of_clusters         = run_parameters['number_of_clusters'        ]
    affinity_metric            = run_parameters['affinity_metric']
    linkage_criterion          = run_parameters['linkage_criterion']

    spreadsheet_name_full_path = run_parameters['spreadsheet_name_full_path']

    spreadsheet_df             = kn.get_spreadsheet_df(spreadsheet_name_full_path)
    spreadsheet_mat            = spreadsheet_df.as_matrix()
    number_of_samples          = spreadsheet_mat.shape[1]

    labels                     = perform_link_hclust( spreadsheet_mat.T
                                                    , number_of_clusters
                                                    , nearest_neighbors
                                                    , affinity_metric
                                                    , linkage_criterion )
        
    sample_names               = spreadsheet_df.columns

    save_final_samples_clustering        (sample_names  , labels, run_parameters)
    save_spreadsheet_and_variance_heatmap(spreadsheet_df, labels, run_parameters)

    return labels

def labels_to_hmat(labels, number_of_clusters):
    """ Convert labels in sampled data to a binary matrix for consensus clustering methods.

    Args:
        labels:             1 x sample size labels array
        number_of_clusters: number of clusters

    Output:
        h_mat:              binary matrix number_of_clusters x sample size
    """
    col    = labels.shape[0]
    mtx    = csr_matrix((np.ones(col), (labels, np.arange(col))), shape=(number_of_clusters, col))
    return mtx.toarray()

def save_final_samples_clustering(sample_names, labels, run_parameters):
    """ write .tsv file that assings a cluster number label to the sample_names.

    Args:
        sample_names: (unique) data identifiers.
        labels: cluster number assignments.
        run_parameters: write path (run_parameters["results_directory"]).

    Output:
        samples_labeled_by_cluster_{method}_{timestamp}_viz.tsv
        phenotypes_labeled_by_cluster_{method}_{timestamp}_viz.tsv
    """

    
    cluster_labels_df  = kn.create_df_with_sample_labels(sample_names, labels)
    cluster_mapping_full_path = get_output_file_name(run_parameters, 'samples_label_by_cluster', 'viz')
    cluster_labels_df.to_csv(cluster_mapping_full_path, sep='\t', header=None, float_format='%g')
    

    if 'phenotype_name_full_path' in run_parameters.keys():
        run_parameters['cluster_mapping_full_path'] = cluster_mapping_full_path
        cluster_eval.clustering_evaluation(run_parameters)

def save_spreadsheet_and_variance_heatmap(spreadsheet_df, labels, run_parameters):
    """ save the full rows by columns spreadsheet.
        Also save variance in separate file.
    Args:
        spreadsheet_df: the dataframe as processed
        run_parameters: with keys for "results_directory", "method", (optional - "top_number_of_rows")

    Output:
        rows_by_samples_heatmp_{method}_{timestamp}_viz.tsv
        rows_averages_by_cluster_{method}_{timestamp}_viz.tsv
        top_rows_by_cluster_{method}_{timestamp}_download.tsv
    """

    top_number_of_rows = run_parameters['top_number_of_rows']
    clusters_df        = spreadsheet_df
    cluster_ave_df     = pd.DataFrame({i: spreadsheet_df.iloc[:, labels == i].mean(axis=1) for i in np.unique(labels)})

    col_labels  = []
    for cluster_number in np.unique(labels):
        col_labels.append('Cluster_%d'%(cluster_number))
    cluster_ave_df.columns = col_labels

    clusters_variance_df  = pd.DataFrame( clusters_df.var(axis=1)
                                        , columns=['variance']                  )
    top_number_of_rows_df = pd.DataFrame( data=np.zeros((cluster_ave_df.shape))
                                        , columns=cluster_ave_df.columns
                                        , index=cluster_ave_df.index.values     )

    for sample in top_number_of_rows_df.columns.values:
        top_index                                                           = np.argsort(cluster_ave_df[sample].values)[::-1]
        top_number_of_rows_df[sample].iloc[top_index[0:top_number_of_rows]] = 1

    file_name_1 = get_output_file_name(run_parameters, 'rows_by_columns_heatmap' , 'viz')
    file_name_2 = get_output_file_name(run_parameters, 'rows_averages_by_cluster', 'viz')
    file_name_3 = get_output_file_name(run_parameters, 'rows_variance',            'viz')
    file_name_4 = get_output_file_name(run_parameters, 'top_rows_by_cluster', 'download')

    clusters_df.to_csv          (file_name_1, sep='\t', float_format='%g')
    cluster_ave_df.to_csv       (file_name_2, sep='\t', float_format='%g')
    clusters_variance_df.to_csv (file_name_3, sep='\t', float_format='%g')
    top_number_of_rows_df.to_csv(file_name_4, sep='\t', float_format='%g')


def save_consensus_clustering(consensus_matrix, sample_names, labels, run_parameters):
    """ write the consensus matrix as a dataframe with sample_names column lablels
        and cluster labels as row labels.

    Args:
        consensus_matrix: sample_names x sample_names numerical matrix.
        sample_names: data identifiers for column names.
        labels: cluster numbers for row names.
        run_parameters: path to write to consensus_data file (run_parameters["results_directory"]).

    Output:
        consensus_matrix_{method}_{timestamp}_viz.tsv
        silhouette_average_{method}_{timestamp}_viz.tsv
    """
    out_df = pd.DataFrame(data=consensus_matrix, columns=sample_names, index=sample_names)

    file_name_1 = get_output_file_name(run_parameters, 'consensus_matrix'  , 'viz')
    file_name_2 = get_output_file_name(run_parameters, 'silhouette_average', 'viz')

    out_df.to_csv(file_name_1, sep='\t', float_format='%g')

    n_labels = len(set(labels))
    n_samples= len(sample_names)

    if (n_labels < 2) or (n_labels > n_samples-1):
        silhouette_average = 1.0
    else:
        silhouette_average = silhouette_score(consensus_matrix, labels)

    silhouette_score_string = 'silhouette number of clusters = %d, corresponding silhouette score = %g' % (
        n_labels, silhouette_average)

    with open(file_name_2,'w') as fh:
        fh.write(silhouette_score_string)


def get_output_file_name(run_parameters, prefix_string, suffix_string='', type_suffix='tsv'):
    """ get the full directory / filename for writing
    Args:
        run_parameters: dictionary with keys: "results_directory", "method" and "correlation_measure"
        prefix_string:  the first letters of the ouput file name
        suffix_string:  the last letters of the output file name before '.tsv'

    Returns:
        output_file_name:   full file and directory name suitable for file writing
    """
    output_file_name = os.path.join(run_parameters["results_directory"], prefix_string + '_' + run_parameters['method'])
    output_file_name = kn.create_timestamped_filename(output_file_name) + '_' + suffix_string + '.' + type_suffix

    return output_file_name

def update_tmp_directory(run_parameters, tmp_dir):
    ''' Update tmp_directory value in rum_parameters dictionary

    Args:
        run_parameters: run_parameters as the dictionary config
        tmp_dir: temporary directory prefix subjected to different functions

    Returns:
        run_parameters: an updated run_parameters

    '''
    if (run_parameters['processing_method'] == 'distribute'):
        run_parameters["tmp_directory"] = kn.create_dir(run_parameters['cluster_shared_volumn'], tmp_dir)
    else:
        run_parameters["tmp_directory"] = kn.create_dir(run_parameters["run_directory"], tmp_dir)

    return run_parameters
