# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install essential build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libssl-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Download and build CMake 3.31.3 from source
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.3/cmake-3.31.3-Linux-x86_64.sh \
-q -O /tmp/cmake-install.sh \
&& chmod u+x /tmp/cmake-install.sh \
&& mkdir /opt/cmake-3.31.3 \
&& /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.31.3 \
&& rm /tmp/cmake-install.sh \
&& ln -s /opt/cmake-3.31.3/bin/* /usr/local/bin

# Verify CMake installation
RUN cmake --version

# Copy project files
COPY . .

# Create build directory and run CMake
RUN cmake -B build -S . && \
    cmake --build build

# Set default command to run the executable
CMD ["./build/cuda_app"]
