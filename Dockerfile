FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

# RUN useradd -m prover
# USER prover

ENV DEBIAN_FRONTEND noninteractive

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    software-properties-common build-essential autotools-dev \
    nfs-common pdsh \
    cmake g++ gcc \
    tmux emacs less unzip \
    htop iftop iotop ca-certificates openssh-client openssh-server \
    rsync iputils-ping net-tools sudo \
    llvm-dev \
    tzdata \
    vim \
    git \
    pipx \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    pigz \
    lz4 \
    zstd \
    numactl \
    libaio-dev \
    libc6-dev

##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
        apt-get update && \
        apt-get install -y git && \
        git --version

##############################################################################
# Client Liveness & Uncomment Port 22 for SSH Daemon
##############################################################################
# Keep SSH client alive from server side
RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
RUN cp /etc/ssh/sshd_config ${STAGE_DIR}/sshd_config && \
        sed "0,/^#Port 22/s//Port 22/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config

# from https://github.com/microsoft/DeepSpeed/blob/master/docker/Dockerfile
##############################################################################
# Mellanox OFED
##############################################################################
ENV MLNX_OFED_VERSION=4.9-7.1.0.0
RUN apt-get install -y libnuma-dev
RUN cd ${STAGE_DIR} && \
        wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-${MLNX_OFED_VERSION}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64.tgz | tar xzf - && \
        cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64 && \
        ./mlnxofedinstall --user-space-only --without-fw-update --all -q && \
        cd ${STAGE_DIR} && \
        rm -rf ${STAGE_DIR}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64*

##############################################################################
# nv_peer_mem
##############################################################################
ENV NV_PEER_MEM_VERSION=1.2
ENV NV_PEER_MEM_TAG=${NV_PEER_MEM_VERSION}-0
RUN mkdir -p ${STAGE_DIR} && \
        git clone https://github.com/Mellanox/nv_peer_memory.git --branch ${NV_PEER_MEM_TAG} ${STAGE_DIR}/nv_peer_memory && \
        cd ${STAGE_DIR}/nv_peer_memory && \
        ./build_module.sh && \
        cd ${STAGE_DIR} && \
        tar xzf ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_VERSION}.orig.tar.gz && \
        cd ${STAGE_DIR}/nvidia-peer-memory-${NV_PEER_MEM_VERSION} && \
        apt-get update && \
        apt-get install -y dkms && \
        dpkg-buildpackage -us -uc && \
        dpkg -i ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_TAG}_all.deb

##############################################################################
# OPENMPI
##############################################################################
ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.6
RUN cd ${STAGE_DIR} && \
        wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
        cd openmpi-${OPENMPI_VERSION} && \
        ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
        make -j"$(nproc)" install && \
        ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
        # Sanity check:
        test -f /usr/local/mpi/bin/mpic++ && \
        cd ${STAGE_DIR} && \
        rm -r ${STAGE_DIR}/openmpi-${OPENMPI_VERSION}
ENV PATH=/usr/local/mpi/bin:${PATH} \
        LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}
# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
        echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
        echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
        chmod a+x /usr/local/mpi/bin/mpirun

## DeepSpeed Ops Compatibility
ENV LDFLAGS="-Wl,--no-as-needed -ldl -lrt"
RUN git clone https://github.com/NVIDIA/cutlass.git && mv cutlass .cutlass
ENV CUTLASS_PATH=$HOME/.cutlass

##############################################################################
# Python
##############################################################################
ENV PYTHON_VERSION=3.11
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt update \
    && apt install -y --no-install-recommends python3.11 \
    && apt install --reinstall -y python3.11-distutils python3.11-venv \
    && rm -f /usr/bin/python \
    && rm -f /usr/bin/python3 \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && ln -s /usr/bin/python3.11 /usr/bin/python3 \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && rm get-pip.py \
    && pip install --upgrade pip

# Print python an pip version
RUN python -V && pip -V

# Install pipx and poetry using pipx
# RUN pip install --user pipx
# ENV PATH="/root/.local/bin:$PATH"
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry config virtualenvs.options.system-site-packages true

RUN apt-get update && apt-get remove -y python3-distutils
RUN pip install --ignore-installed --no-cache-dir PyYAML==6.0

COPY pyproject.toml /workspace/pyproject.toml
COPY README.md /workspace/README.md

ENV NO_PROXY='' HTTP_PROXY='' HTTPS_PROXY=''
ENV no_proxy='' http_proxy='' https_proxy=''

RUN poetry install

RUN python -c "import deepspeed; print(deepspeed.__version__)"
RUN ds_report

RUN pip install flash-attn==2.7.3
RUN python -c "import flash_attn; print(flash_attn.__version__)"

RUN apt-get install -y \
    bc \
    libpython3.11-dev \
    dnsutils \
    iputils-ping \
    ssh
