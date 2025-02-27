FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libssl-dev \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.3/cmake-3.31.3-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.31.3 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.31.3 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.31.3/bin/* /usr/local/bin

RUN cmake --version

RUN git clone https://github.com/Mayukh-Banik/CudaLearn.git

WORKDIR /CudaLearn

RUN git submodule update --init --recursive

RUN cmake --preset default && cmake --build --preset --default
RUN cmake --preset python-package && cmake --build --python-package

RUN python3 -m venv venv && source venv/bin/activate
RUN pip install -e .

CMD ["./build/tests/tests"]
