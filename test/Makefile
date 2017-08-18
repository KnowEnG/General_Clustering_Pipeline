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
run_hclustering:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_1_hclust.yml

run_kmeans:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file BENCHMARK_2_kmeans.yml


run_all_methods: run_hclustering run_kmeans

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


