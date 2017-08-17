### How To Generate _html_ Documentation After Cloning the Repository

* Install sphinx, at the command prompt type:
    * >> pip install sphinx
    
* Change to the General_Clustering_Pipeline/docs directory in your cloned code.
* Generate the intermediate (.rst) files at the command prompt:
    * >> sphinx-apidoc -f -o source/ ../src/
    
* Use the ./docs/Makefile to generate the _html_ code:
    * >> make html

* In your browser menu use File - Open File... to bookmark the ./docs/build/html/index.html homepage.
* that page has links to search, index and functions listing pages with links to view the source code.
