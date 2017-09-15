* * * 
## How to verify this pipeline installation on your computer
Use verification testing to assure that the runtime environment and the current version produce the expected output using this repository's data.
* * * 
### 1. Clone the General_Clustering_Pipeline Repo
```
 git clone https://github.com/KnowEnG-Research/General_Clustering_Pipeline.git
```
### 2. Install the following (Ubuntu or Linux)
  ```
 pip3 install pyyaml
 pip3 install knpackage
 pip3 install scipy==0.19.1
 pip3 install numpy==1.11.1
 pip3 install pandas==0.18.1
 pip3 install matplotlib==1.4.2
 pip3 install scikit-learn==0.17.1
 
 apt-get install -y python3-pip
 apt-get install -y libfreetype6-dev libxft-dev
 apt-get install -y libblas-dev liblapack-dev libatlas-base-dev gfortran
```
### 3. Change directory to General_Clustering_Pipeline
```
cd General_Clustering_Pipeline/test
```
### 4. Start the verification test from the command line
```
make verification_tests
```
### 5. The output files will be compared with the General_Clustering_Pipeline/data/verification/... data
* Each Benchmark will report PASS or FAIL and list the names of files producing differences (if any).
* Note that the files generated will be erased after each Benchmark test.
