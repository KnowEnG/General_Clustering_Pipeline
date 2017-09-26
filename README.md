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

## How to install this pipeline
Install prerequisite packages, and clone the repo:
```
apt-get install -y python3-pip libfreetype6-dev libxft-dev libblas-dev liblapack-dev libatlas-base-dev gfortran
pip3 install pyyaml knpackage scipy==0.19.1 numpy==1.11.1 pandas==0.18.1 matplotlib==1.4.2 scikit-learn==0.17.1 
git clone https://github.com/KnowEnG-Research/General_Clustering_Pipeline.git
```

## How to run this pipeline with its bundled data
Make the directory `./run_dir/` to store the output files:
```
cd General_Clustering_Pipeline/test
make env_setup
```

Run one of these commands:

| **Command**                     | **Option**                                 | 
|:------------------------------- |:-------------------------------------------| 
| `make run_kmeans_binary`          | Clustering with k-means                    |
| `make run_kmeans_continuous`      |                                            |
| `make run_hclust_binary`          | Hierarchical Clustering                    |
| `make run_hclust_continuous`      |                                            |
| `make run_link_hclust_binary`     | Hierarchical linkage Clustering            |
| `make run_link_hclust_continuous` |                                            |
| `make run_cc_kmeans_binary`       | Consensus Clustering with k-means          |
| `make run_cc_kmeans_continuous`   |                                            |
| `make run_cc_hclust_binary`       | Consensus Hierarchical Clustering          |
| `make run_cc_hclust_continuous`   |                                            |
| `make run_cc_link_hclust_binary`  | Consensus Hierarchical linkage Clustering  |


## How to run this pipeline with your own data
### Create your run directory and results directory
```
mkdir run_dir
cd run_dir
mkdir results
```

### Create and modify the run_parameters file (in YAML format)
- Copy an example such as `General_Clustering_Pipeline/data/run_files/zTEMPLATE_*.yml`.

- In your copy, change `processing_method` to `serial` or `parallel`, depending on your machine.

- Set the data file targets to the files you want to run, and the parameters as appropriate for your data.

| **Key**                    | **Example value**             | **Description**                               |
| -------------------------  | ----------------------------- | --------------------------------------------- |
| method                     | hclust, cc_hclust,...         | Choose clustering method                      |
| spreadsheet_name_full_path | data/spreadsheets/foo.gxc.tsv | Path and file name of user-supplied gene sets |
| results_directory          | run_dir/results               | Directory to save the output files            |
| number_of_clusters         | 3                             | Estimated number of clusters                  |

<!-- Also: nearest_neighbors, top_number_of_rows, cluster_ip_address, cluster_shared_ram, cluster_shared_volumn -->


  * Run
  `python3 ../src/general_clustering.py -run_directory ./run_dir -run_file zTEMPLATE_cc_net_nmf.yml`

## Description of Output files saved in results directory

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
