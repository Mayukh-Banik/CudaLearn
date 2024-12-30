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

Plain integer example

Tensor* VectorAdd_double_from_int8_and_double(Tensor a, Tensor b)
{
    Tensor* c = new Tensor(std::max(a.Ndim, b.NDim), std::max(a.)
}