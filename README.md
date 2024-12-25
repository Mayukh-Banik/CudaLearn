# Build the Docker image
docker build -t cuda_app .

# Run with NVIDIA runtime
docker run --gpus all cuda_app