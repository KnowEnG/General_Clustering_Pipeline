"""
sobh@illinois.edu

"""

import os
import filecmp
import time

verif_dir = '../data/verification'

def verify_benchmark_1():

    print('\n\tBENCHMARK_1_SC_nmf.yml')
    t0 = time.time()
    os.system('python3 ../src/general_clustering.py -run_directory ./run_dir -run_file BENCHMARK_1_hclustering.yml')

    verif_file = os.path.join(verif_dir, 'general_label_by_cluster_hclustering_BENCHMARK_1.tsv')

    results_dir = '../test/run_dir/results'
    results_prefix = 'general_label_by_cluster_nmf_'
    results_file_list = os.listdir(results_dir)

    verifiable_files_list = []
    for f in results_file_list:
        if f[0:len(results_prefix)] == results_prefix:
            verifiable_files_list.append(os.path.join(results_dir, f))
    if len(verifiable_files_list) > 0:
        for veri_file_name in verifiable_files_list:
            if filecmp.cmp(veri_file_name, verif_file) == True:
                print(veri_file_name, '\n\t\t BENCHMARK_1 Verification Success', time.time() - t0)
            else:
                print(veri_file_name, 'Differs: BENCHMARK_1 Verification Fails')
    else:
        print(verif_file, 'no file name match found')


def verify_benchmark_2():
    print('\n\tBENCHMARK_2_kmeans.yml')
    t0 = time.time()
    os.system('python3 ../src/general_clustering.py -run_directory ./run_dir -run_file BENCHMARK_2_kmeans.yml')

    verif_file = os.path.join(verif_dir, 'general_label_by_cluster_net_nmf_BENCHMARK_2.tsv')

    results_dir = '../test/run_dir/results'
    results_prefix = 'general_label_by_cluster_net_nmf_'
    results_file_list = os.listdir(results_dir)

    verifiable_files_list = []
    for f in results_file_list:
        if f[0:len(results_prefix)] == results_prefix:
            verifiable_files_list.append(os.path.join(results_dir, f))
    if len(verifiable_files_list) > 0:
        for veri_file_name in verifiable_files_list:
            if filecmp.cmp(veri_file_name, verif_file) == True:
                print(veri_file_name, '\n\t\t BENCHMARK_2 Verification Success', time.time() - t0)
            else:
                print(veri_file_name, 'Differs: BENCHMARK_2 Verification Fails')
    else:
        print(verif_file, 'no file name match found')


def main():
    os.system('make env_setup')
    verify_benchmark_1()
    verify_benchmark_2()
    print('\n')


if __name__ == "__main__":
    main()
