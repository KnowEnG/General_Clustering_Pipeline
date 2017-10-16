"""
@author: The KnowEnG dev team
"""
import os
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
import knpackage.toolbox as kn


class ColumnType(Enum):
    """Two categories of phenotype traits.
    """
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


def combine_phenotype_data_and_clustering(run_parameters):
    """This is to insert the sample clusters column into the phenotype dataframe.
    Returns:
        phenotype_df: phenotype dataframe with the first column as sample clusters.
    """
    phenotype_df = kn.get_spreadsheet_df(run_parameters['phenotype_name_full_path'])
    phenotype_df.insert(0, 'Cluster_ID', np.nan) # pylint: disable=no-member
    cluster_labels_df = pd.read_csv(
        run_parameters['cluster_mapping_full_path'], index_col=0, header=None, sep='\t')
    cluster_labels_df.columns = ['Cluster_ID']
    common_samples = kn.find_common_node_names(phenotype_df.index, cluster_labels_df.index)
    phenotype_df.loc[common_samples, 'Cluster_ID'] = cluster_labels_df.loc[common_samples, 'Cluster_ID'] # pylint: disable=no-member
    return phenotype_df


def run_post_processing_phenotype_clustering_data(cluster_phenotype_df, threshold):
    """This is the clean up function of phenotype data with nans removed.
    Parameters:
        cluster_phenotype_df: phenotype dataframe with the first column as sample clusters.
        threshold: threshold to determine which phenotype to remove.
    Returns:
        output_dict: dictionary with keys to be categories of phenotype data and values
        to be a list of related dataframes.
    """
    from collections import defaultdict

    output_dict = defaultdict(list)
    fail_df = pd.DataFrame(index=['Measure', 'Trait_length_after_dropna', \
        'Sample_number_after_dropna', 'chi/fval', 'pval', 'SUCCESS/FAIL', 'Comments'])


    for column in cluster_phenotype_df:
        if column == 'Cluster_ID':
            continue
        cur_df = cluster_phenotype_df[['Cluster_ID', column]].dropna(axis=0)
        if cur_df.empty:
            fail_df[column] \
            = [np.nan, 0, 0, np.nan, 1, 'FAIL', 'Input phenotype is empty']
        if not cur_df.empty:
            if cur_df[column].dtype == object:
                cur_df_lowercase = cur_df.apply(lambda x: x.astype(str).str.lower())
            else:
                cur_df_lowercase = cur_df

            num_uniq_value = len(cur_df_lowercase[column].unique())
            num_total = len(cur_df_lowercase[column])

            if num_uniq_value == 1:
                fail_df[column] \
                = [np.nan, 1, num_total, np.nan, 1, 'FAIL', 'Number of unique trait is one']
                continue
            if cur_df_lowercase[column].dtype == object and num_uniq_value > threshold:
                fail_df[column] \
                = [np.nan, num_uniq_value, num_total, np.nan, 1, \
                'FAIL', 'Number of unique categorical trait is not below threshold']
                continue
            if num_uniq_value > threshold:
                classification = ColumnType.CONTINUOUS
            else:
                classification = ColumnType.CATEGORICAL
            output_dict[classification].append(cur_df_lowercase)
    return output_dict, fail_df


def f_oneway(phenotype_df):
    """ Perform a f_oneway test and report the results.
    Parameters:
        phenotype_df: dataframe with two columns with clusters and phenotype trait values.
        ret: result of the phenotype dataframe.
    """
    uniq_trait = np.unique(phenotype_df.values[:, 1].reshape(-1))
    uniq_cluster = np.unique(phenotype_df.values[:, 0])
    if len(uniq_cluster) == 1:
        comment = 'The number of clusters is one'
        return ['f_oneway', len(uniq_trait), phenotype_df.shape[0], np.nan, 1, 'FAIL', comment]

    groups = []
    uniq_cm_vals = sorted(set(phenotype_df.values[:, 0]))

    phenotype_name = phenotype_df.columns.values[1]
    for i in uniq_cm_vals:
        groups.append(
            phenotype_df.loc[phenotype_df['Cluster_ID'] == i, phenotype_name].values.tolist())

    fval, pval = stats.f_oneway(*groups)
    ret = ['f_oneway', len(uniq_trait), phenotype_df.shape[0], fval, pval, 'SUCCESS', np.nan]
    return ret


def chisquare(phenotype_df):
    """ Perform a chi-square test and report the results.
    Parameters:
        phenotype_df: dataframe with two columns with clusters and phenotype trait values.
        ret: result of the phenotype dataframe.
    """
    uniq_category = np.unique(phenotype_df.values[:, 1])
    uniq_cluster = np.unique(phenotype_df.values[:, 0])
    num_clusters = len(uniq_cluster)
    num_phenotype = len(uniq_category)
    phenotype_name = phenotype_df.columns.values[1]
    phenotype_val_dict = dict(zip(uniq_category, range(num_phenotype)))
    cluster_dict = dict(zip(uniq_cluster, range(num_clusters)))

    cont_table = np.zeros((num_clusters, num_phenotype))

    for sample in phenotype_df.index:
        clus = cluster_dict[phenotype_df.loc[sample, 'Cluster_ID']]
        trt = phenotype_val_dict[phenotype_df.loc[sample, phenotype_name]]  # pylint: disable=no-member
        cont_table[clus, trt] += 1

    chi, pval, dof, expected = stats.chi2_contingency(cont_table)
    ret = ['chisquare', num_phenotype, phenotype_df.shape[0], chi, pval, 'SUCCESS', np.nan]
    return ret


def clustering_evaluation(run_parameters):
    """ Run clustering evaluation on the whole dataframe of phenotype data.
    Save the results to tsv file.
    """
    cluster_phenotype_df = combine_phenotype_data_and_clustering(run_parameters)
    output_dict, fail_df = run_post_processing_phenotype_clustering_data(cluster_phenotype_df, run_parameters['threshold'])

    result_df = pd.DataFrame(index=['Measure', 'Trait_length_after_dropna', \
        'Sample_number_after_dropna', 'chi/fval', 'pval', 'SUCCESS/FAIL', 'Comments'])

    for key, df_list in output_dict.items():
        if key == ColumnType.CATEGORICAL:
            for item in df_list:
                phenotype_name = item.columns.values[1]
                result_df[phenotype_name] = chisquare(item)
        else:
            for item in df_list:
                phenotype_name = item.columns.values[1]
                result_df[phenotype_name] = f_oneway(item)

    file_name = kn.create_timestamped_filename("clustering_evaluation_result", "tsv")
    file_path = os.path.join(run_parameters["results_directory"], file_name)
    result_df = pd.concat([result_df, fail_df], axis=1)
    result_df = result_df.T.sort_index()
    result_df.to_csv(file_path, header=True, index=True, sep='\t', na_rep='NA')
