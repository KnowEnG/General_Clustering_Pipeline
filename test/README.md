## How to verify this pipeline installation on your computer
Use verification testing to assure that the runtime environment and the current version produce the expected output using this repository's data.

### Install the prerequisite packages and clone the repo
```
apt-get install -y python3-pip libfreetype6-dev libxft-dev libblas-dev liblapack-dev libatlas-base-dev gfortran
pip3 install pyyaml knpackage scipy==0.19.1 numpy==1.11.1 pandas==0.18.1 matplotlib==1.4.2 scikit-learn==0.17.1
git clone https://github.com/KnowEnG-Research/General_Clustering_Pipeline.git
```

### Run the test
```
cd General_Clustering_Pipeline/test
make verification_tests
```

### The output files will be compared with the data in `General_Clustering_Pipeline/data/verification/`
* Each Benchmark will report PASS or FAIL and list any files producing differences.
* Note that the files generated will be erased after each Benchmark test.
