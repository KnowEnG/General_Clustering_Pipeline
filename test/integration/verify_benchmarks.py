"""
sobh@illinois.edu

"""

import os
import filecmp

verification_dir = '../data/verification'
results_dir      = '../test/run_dir/results'

def verify_benchmark(BENCHMARK_name,BENCHMARK_YML) :

    run_command  = 'python3 ../src/general_clustering.py -run_directory ./run_dir -run_file ' + BENCHMARK_YML
    os.system(run_command)

    All_files_in_results_dir = os.listdir(results_dir)

    for f in All_files_in_results_dir:
        if BENCHMARK_name in f :
            RESULT    = os.path.join(results_dir,      f             )
            BENCHMARK = os.path.join(verification_dir, BENCHMARK_name+'.tsv')
            if filecmp.cmp(RESULT, BENCHMARK) == True:
                print(BENCHMARK, '______ PASS ______' )
            else:
                print(BENCHMARK, '****** FAIL ******' )

def main():
    BENCHMARK = {'kmeans'    : [ 
                                 'BENCHMARK_1_kmeans_binary.yml'
                               , 'rows_averages_by_cluster_kmeans'
                               , 'rows_by_columns_heatmap_kmeans'
                               , 'rows_variance_kmeans'
                               , 'samples_label_by_cluster_kmeans'
                               , 'top_rows_by_cluster_kmeans'
                               ] 
               ,'hclust'     : [  
                                 'BENCHMARK_3_hclust_binary.yml'
                               , 'rows_averages_by_cluster_hclust'
                               , 'rows_by_columns_heatmap_hclust'
                               , 'rows_variance_hclust'
                               , 'samples_label_by_cluster_hclust'
                               , 'top_rows_by_cluster_hclust'
                               ] 
               ,'link_hclust': [  
                                 'BENCHMARK_5_link_hclust_binary.yml'
                               , 'rows_averages_by_cluster_link_hclust'
                               , 'rows_by_columns_heatmap_link_hclust'
                               , 'rows_variance_link_hclust'
                               , 'samples_label_by_cluster_link_hclust'
                               , 'top_rows_by_cluster_link_hclust'
                               ]
               ,'cc_kmeans'  : [  
                                 'BENCHMARK_7_cc_kmeans_binary.yml'
                               , 'consensus_matrix_cc_kmeans'
                               , 'rows_averages_by_cluster_cc_kmeans'
                               , 'rows_by_columns_heatmap_cc_kmeans'
                               , 'rows_variance_cc_kmeans'
                               , 'samples_label_by_cluster_cc_kmeans'
                               , 'silhouette_average_cc_kmeans'
                               , 'top_rows_by_cluster_cc_kmeans'
                               ]

               ,'cc_hclust'  : [  
                                 'BENCHMARK_9_cc_hclust_binary.yml'
                               , 'consensus_matrix_cc_hclust'
                               , 'rows_averages_by_cluster_cc_hclust'
                               , 'rows_by_columns_heatmap_cc_hclust'
                               , 'rows_variance_cc_hclust'
                               , 'samples_label_by_cluster_cc_hclust'
                               , 'silhouette_average_cc_hclust'
                               , 'top_rows_by_cluster_cc_hclust'
                               ]
                }

    os.system('make env_setup')
    for key in BENCHMARK.keys(): 
        BENCHMARK_list = BENCHMARK[key]
        BENCHMARK_YML  = BENCHMARK_list[0]
        for BENCHMARK_name in BENCHMARK_list[1:] :
            verify_benchmark(BENCHMARK_name,BENCHMARK_YML)
            os.system('rm ./run_dir/results/*')

if __name__ == "__main__":
    main()
