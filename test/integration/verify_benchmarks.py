"""
sobh@illinois.edu

"""

import filecmp
import os
import time

verification_dir = '../data/verification/'
results_dir = '../test/run_dir/results'


def verify_benchmark(option, algo_name, BENCHMARK_name_list, BENCHMARK_YML):
    run_command = 'python3 ../src/general_clustering.py -run_directory ./run_dir -run_file ' + BENCHMARK_YML
    os.system(run_command)

    All_files_in_results_dir = os.listdir(results_dir)

    num_failed_tests = 0
    num_succeed_tests = 0
    for f in All_files_in_results_dir:
        for BENCHMARK_name in BENCHMARK_name_list:
            if BENCHMARK_name in f:
                RESULT = os.path.join(results_dir, f)
                BENCHMARK = os.path.join(verification_dir, option, algo_name, BENCHMARK_name + '.tsv')
                if filecmp.cmp(RESULT, BENCHMARK) == True:
                    num_succeed_tests += 1
                    print(BENCHMARK, '______ PASS ______')
                else:
                    num_failed_tests += 1
                    print(BENCHMARK, '****** FAIL ******')
    return num_succeed_tests, num_failed_tests


def main():
    BENCHMARK = {
        'binary': {
            'kmeans_evaluation': [
                'BENCHMARK_13_kmeans_binary_evaluation.yml',
                'rows_averages_by_cluster_kmeans',
                'rows_by_columns_heatmap_kmeans',
                'rows_variance_kmeans',
                'samples_label_by_cluster_kmeans',
                'top_rows_by_cluster_kmeans',
                'clustering_evaluation_result'
            ]
        }
    }

    os.system('make env_setup')
    start_time = time.time()
    total_success, total_failure = 0, 0
    for option in BENCHMARK.keys():
        for key in BENCHMARK[option].keys():
            BENCHMARK_list = BENCHMARK[option][key]
            BENCHMARK_YML = BENCHMARK_list[0]
            print()
            print("INFO: Running test ", "./run_dir/results/" + BENCHMARK_YML)
            # for BENCHMARK_name in BENCHMARK_list[1:]:
            num_succeed_tests, num_failed_tests = verify_benchmark(option, key, BENCHMARK_list[1:], BENCHMARK_YML)
            total_success += num_succeed_tests
            total_failure += num_failed_tests
            os.system('rm ./run_dir/results/*')
    end_time = time.time()
    print()
    print("Ran {} tests in {}s".format(total_success + total_failure, end_time - start_time))
    if (total_failure == 0):
        print("OK")
        print()
    else:
        print("FAILED(errors={})".format(total_failure))


if __name__ == "__main__":
    main()
