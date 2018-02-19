FROM knowengdev/base_image:09_19_2017
LABEL Xi Chen="xichen24@illinois.edu" \
      Jing Ge="jingge2@illinois.edu" \
      Dan Lanier="lanier4@illinois.edu" \
      Nahil Sobh="sobh@illinois.edu"

ENV SRC_LOC /home

# Install the latest knpackage
RUN pip3 install -I knpackage

# Copy source code to docker container
COPY src ${SRC_LOC}/src
COPY test ${SRC_LOC}/test
COPY data ${SRC_LOC}/data
COPY docs ${SRC_LOC}/docs
COPY LICENSE ${SRC_LOC}
COPY README.md ${SRC_LOC}

# Set up working directory
WORKDIR ${SRC_LOC}
