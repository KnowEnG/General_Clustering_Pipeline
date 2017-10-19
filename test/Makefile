SCRIPT = ../src/general_clustering.py
RUN_DIR = ./run_dir
RUN_FILES = ../data/run_files
RESULTS_DIR = $(RUN_DIR)/results
INPUT_DATA_DIR = ../data
UNIT_TEST_TEMP_DIR = ./unit/tmp

RMR = rm -rf
MKDIR_P = mkdir -p

# ----------------------------------------------------------------
# Benchmarks setup Section                                       -
# ----------------------------------------------------------------
env_setup: create_run_dir copy_run_files

create_run_dir:
	$(MKDIR_P) $(RESULTS_DIR) 

copy_run_files:
	cp $(RUN_FILES)/*.yml $(RUN_DIR) 

clean_dir_recursively:
	$(RMR) $(RUN_DIR)

# ----------------------------------------------------------------
# Large Benchmarks RUN Section                                   -
# ----------------------------------------------------------------
run_all_binary:\
run_kmeans_binary\
run_hclust_binary\
run_link_hclust_binary\
run_cc_kmeans_binary\
run_cc_hclust_binary\
run_cc_link_hclust_binary

run_all_continuous:\
run_kmeans_continuous \
run_hclust_continuous \
run_link_hclust_continuous \
run_cc_kmeans_continuous \
run_cc_hclust_continuous \
run_cc_link_hclust_continuous 

run_kmeans_binary:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_1_kmeans_binary.yml

run_kmeans_continuous:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_2_kmeans_continuous.yml

run_hclust_binary:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_3_hclust_binary.yml

run_hclust_continuous:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_4_hclust_continuous.yml

run_link_hclust_binary:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_5_link_hclust_binary.yml

run_link_hclust_continuous:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_6_link_hclust_continuous.yml

run_cc_kmeans_binary:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_7_cc_kmeans_binary.yml

run_cc_kmeans_continuous:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_8_cc_kmeans_continuous.yml


run_cc_hclust_binary:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_9_cc_hclust_binary.yml

run_cc_hclust_continuous:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_X_cc_hclust_continuous.yml

run_cc_link_hclust_binary:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_11_cc_link_hclust_binary.yml

run_cc_link_hclust_continuous:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_12_cc_link_hclust_continuous.yml

# ----------------------------------------------------------------
# - EVALUATION TESTS RUN SECTION                                 -
# ----------------------------------------------------------------

run_kmeans_binary_eval:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_13_kmeans_binary_evaluation.yml

# ----------------------------------------------------------------
# - VERIFICATION TESTS RUN SECTION                               -
# ----------------------------------------------------------------
verification_tests:
	python3 ./integration/verify_benchmarks.py

# ----------------------------------------------------------------
# - UNIT TESTS RUN SECTION                                       -
# ----------------------------------------------------------------
unit_test_setup:
	$(MKDIR_P) $(UNIT_TEST_TEMP_DIR)

unit_tests:
	cd unit; make all_unit_tests
	
run_benchmark_test:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_1_hclustering.yml

# ----------------------------------------------------------------
# - CLEAN UP SECTION                                             -
# ----------------------------------------------------------------	
unit_test_cleanup:
	$(RMR) $(UNIT_TEST_TEMP_DIR)


