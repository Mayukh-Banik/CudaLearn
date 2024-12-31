A FP-64 Matrix only library.

Aims:
    
    Recreate all NumPy functions in C++.
    
    Bind it with nanobind for Python Usage

    Exporting to NumPy/PyTorch objects using Python Buffer Protocol

    Learning CUDA

    Being able to train ML models, and potentially other tasks

```
git submodule update --init --recursive
cmake -B build -S . [-DENABLE_TESTS=ON/-DENABLE_BINDINGS=ON]
cmake --build build
```

```
cmake -B build -S . -DENABLE_TESTS=ON && cmake --build build && ./build/tests/tests
```

```
docker build -t cudalearn .
docker run --gpus all cudalearn
```