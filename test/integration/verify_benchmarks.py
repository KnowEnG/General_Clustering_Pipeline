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
                print(BENCHMARK, 'PASS' )
            else:
                print(BENCHMARK, 'FAIL' )

def main():
    BENCHMARK = {'kmeans'    : [ 
                                 'BENCHMARK_1_kmeans.yml'
                               , 'rows_averages_by_cluster_kmeans'
                               , 'rows_by_columns_heatmap_kmeans'
                               , 'top_rows_by_cluster_kmeans'
                               , 'rows_variance_kmeans'
                               ] 
               ,'hclust'     : [  
                                 'BENCHMARK_2_hclust.yml'
                               , 'rows_averages_by_cluster_hclust'
                               , 'rows_by_columns_heatmap_hclust'
                               , 'top_rows_by_cluster_hclust'
                               , 'rows_variance_hclust'
                               ] 
               ,'hclust_link': [  
                                 'BENCHMARK_3_hclust_link.yml'
                               , 'rows_averages_by_cluster_hclust_link'
                               , 'rows_by_columns_heatmap_hclust_link'
                               , 'top_rows_by_cluster_hclust_link'
                               , 'rows_variance_hclust_link'
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
