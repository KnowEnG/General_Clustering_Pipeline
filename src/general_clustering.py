"""
Created on Wed Jul 20 14:47:45 2016
@author: The KnowEnG dev team
"""

def hclust(run_parameters):
    '''hierarchical clustering'''
    from general_clustering_toolbox import run_hclust
    run_hclust(run_parameters) 

def kmeans(run_parameters):
    '''kmeans clustering'''
    from general_clustering_toolbox import run_kmeans
    run_kmeans(run_parameters)


SELECT = {
    "hclust":hclust,
    "kmeans":kmeans }

def main():
    """
    This is the main function to perform general clustering
    """
    import sys
    from knpackage.toolbox import get_run_directory_and_file
    from knpackage.toolbox import get_run_parameters
    
    run_directory, run_file = get_run_directory_and_file(sys.argv)
    run_parameters = get_run_parameters(run_directory, run_file)
    SELECT[run_parameters["method"]](run_parameters)

if __name__ == "__main__":
    main()
