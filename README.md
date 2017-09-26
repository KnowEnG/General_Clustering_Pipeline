# KnowEnG's General Clustering Pipeline 
This is the Knowledge Engine for Genomics (KnowEnG), an NIH BD2K Center of Excellence, General Clustering Pipeline.

This pipeline **clusters** a spreadsheet's columns, with various methods:

| **Options**                                      | **Method**                                   | **Parameters** |
| ------------------------------------------------ | ---------------------------------------------| -------------- |
| K-means                                          | K Means                                      | kmeans         |
| hierarchical clustering                          | hierarchical clustering                      | hclust         |
| Linked hierarchical clustering                   | hierarchical clustering constraint           | link_hclust    |
| Bootstrapped hierarchical clustering             | consensus hierarchical clustering            | cc_ hclust     |
| Bootstrapped K-means                             | consensus K Means                            | cc_kmeans      |
| Bootstrapped Linked hierarchical clustering      | consensus linked hierarchical clustering     | cc_link_hclust |     

* * * 
## How to run this pipeline with Our data
* * * 
### 1. Clone the General_Clustering_Pipeline Repo
```
 git clone https://github.com/KnowEnG-Research/General_Clustering_Pipeline.git
```
 
### 2. Install the following, for Linux
```
 apt-get install -y python3-pip libfreetype6-dev libxft-dev libblas-dev liblapack-dev libatlas-base-dev gfortran
 pip3 install pyyaml knpackage scipy==0.19.1 numpy==1.11.1 pandas==0.18.1 matplotlib==1.4.2 scikit-learn==0.17.1 
```

### 3. Change directory to General_Clustering_Pipeline

```
cd General_Clustering_Pipeline
```

### 4. Change directory to test

```
cd test
```
 
### 5. Create a local directory "run_dir" and place all the run files in it
```
make env_setup
```

### 6. Use one of the following "make" commands to select and run a clustering option:


| **Command**                     | **Option**                                 | 
|:------------------------------- |:-------------------------------------------| 
| make run_kmeans_binary          | Clustering with k-means                    |
| make run_kmeans_continuous      |                                            |
| make run_hclust_binary          | Hierarchical Clustering                    |
| make run_hclust_continuous      |                                            |
| make run_link_hclust_binary     | Hierarchical linkage Clustering            |
| make run_link_hclust_continuous |                                            |
| make run_cc_kmeans_binary       | Consensus Clustering with k-means          |
| make run_cc_kmeans_continuous   |                                            |
| make run_cc_hclust_binary       | Consensus Hierarchical Clustering          |
| make run_cc_hclust_continuous   |                                            |
| make run_cc_link_hclust_binary  | Consensus Hierarchical linkage Clustering  |

 
* * * 
## How to run this pipeline with Your data
* * * 

__***Follow steps 1-5 above then do the following:***__

### * Create your run directory

 ```
 mkdir run_dir
 ```

### * Change directory to the run directory

 ```
 cd run_dir
 ```

### * Create your results directory

 ```
 mkdir results
 ```
 
### * Create run_paramters file  (YAML Format)
 ``` 
 Look for examples of run_parameters in the General_Clustering_Pipeline/data/run_files zTEMPLATE_cc_hclust.yml
 ```
### * Modify run_paramters file  (YAML Format)
Change processing_method to one of: serial, parallel depending on your machine.
```
processing_method: serial
```

set the data file targets to the files you want to run, and the parameters as appropriate for your data.


### * Run the General Clustering Pipeline:

  * Update PYTHONPATH enviroment variable
   ``` 
   export PYTHONPATH='../src':$PYTHONPATH    
   ```
   
  * Run
   ```
  python3 ../src/general_clustering.py -run_directory ./run_dir -run_file zTEMPLATE_cc_net_nmf.yml
   ```

* * * 
## Description of "run_parameters" file
* * * 

| **Key**                    | **Value**                    | **Comments**                                   |
| -------------------------  | ---------------------------- | ---------------------------------------------- |
| method                     | **hclust**, **cc_hclust**,...| Choose clustering method                       |
| affinity_metric            | **euclidean**, **manhattan** | Choose clustering affinity                     |
| linkage_criterion          | **ward**, **complete**, **average** | Choose clustering affinity              |
| spreadsheet_name_full_path | directory+spreadsheet_name   |  Path and file name of user supplied gene sets |
| results_directory          | directory                    | Directory to save the output files             |
| number_of_clusters         | 3                            | Estimated number of clusters                   |

spreadsheet_name = ProGENI_rwr20_STExp_GDSC_500.rname.gxc.tsv</br>

* * * 
## Description of Output files saved in results directory
* * * 

* Output files of all  methods save row by col heatmap variances per row with name **row_variance_{method}_{timestamp}_viz.tsv**.</br>

 |  |**variance**|
 | :--------------------: |:--------------------:|
 | **row 1**              |float                 |
 |...                     |...                   |
 | **row m**              | float                |

* Output files of all the methods save row by col heatmap with name **row_by_col_heatmp_{method}_{timestamp}_viz.tsv**.</br>

 |  |**col 1**|...|**col n**|
 | :--------------------: |:--------------------:|:--------------------:|:--------------------:|
 | **row 1**              |float                 |...                   |float                 |
 |...                     |...                   |...                   |...                   |  
 | **row m**              |float                 |...                   |float                 |

 
* Output files of all  methods save col to cluster map with name **col_labeled_by_cluster_{method}_{timestamp}_viz.tsv**.</br>

 |    |**cluster**|
 | :--------------------: |:--------------------:|
 | **col 1**              |int                   |
 |...                     |...                   |
 | **col n**              |int                   |
 
* Output files of all  methods save row scores by cluster with name **row_averages_by_cluster_{method}_{timestamp}_viz.tsv**.</br>

 |  |**cluster 1**|...|**cluster k**|
 | :--------------------: |:--------------------:|:--------------------:|:--------------------:|
 | **row 1**              |float                 |...                   |float                 |
 |...                     |...                   |...                   |...                   |
 | **row m**              |float                 |...                   |float                 |
 
* Output files of all  methods save spreadsheet with top ranked rows per column with name **top_row_by_cluster_{method}_{timestamp}_download.tsv**.</br>

 |  |**cluster 1**|...|**cluster k**|
 | :--------------------: |:--------------------:|:--------------------:|:--------------------:|
 | **row 1**              |1/0                   |...                   |1/0                   |
 |...                     |...                   |...                   |...                   |
 | **row m**              |1/0                   |...                   |1/0                   |
  
* All  methods save **silhouette number of clusters** and **corresponding silhouette score** with name silhouette_average\_{method}\_{timestamp}\_viz.tsv.</br>
 ```
 File Example: 
 silhouette number of clusters = 3, corresponding silhouette score = 1
 ```
