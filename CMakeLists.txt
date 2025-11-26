cmake_minimum_required(VERSION 3.15)
project(CompressND LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ADIOS2 REQUIRED COMPONENTS CXX MPI)

find_package(MPI REQUIRED)

add_executable(compress_mpi compress_mpi.cpp)
target_link_libraries(compress_mpi
    PRIVATE
        MPI::MPI_CXX
        adios2::adios2
)



add_executable(decompress_mpi decompress_mpi.cpp)
target_link_libraries(decompress_mpi
    PRIVATE
        MPI::MPI_CXX
        adios2::adios2
)

