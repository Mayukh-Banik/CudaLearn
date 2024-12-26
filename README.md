git submodule update --init --recursive
cmake -B build -S . [-DENABLE_TESTS=ON/-DENABLE_BINDINGS=ON]
cmake --build build