FROM knowengdev/base_image:09_19_2017
MAINTAINER Nahil Sobh <sobh@illinois.edu>

ENV SRC_LOC /home

# Install the latest knpackage
RUN pip3 install -I knpackage

# Clone from github
RUN git clone https://github.com/KnowEnG/General_Clustering_Pipeline.git ${SRC_LOC} 

# Set up working directory
WORKDIR ${SRC_LOC}
