# for cscv project
project(cscv)
cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set icpc if unset
if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "g++")
endif()
# set(CMAKE_CXX_COMPILER "g++")

# if icpc
if (CMAKE_CXX_COMPILER MATCHES "icpc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -xHost -qmkl -ltbb -qopenmp -std=c++17 -g -diag-disable=10441")
# else, for g++
else()
    # use intel omp
    set(MKL_LD "-Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl -lmkl_gnu_thread -liomp5")
    set(CMAKE_CXX_FLAGS "-O3 -march=native -g -fopenmp -std=c++17 -fno-strict-aliasing -Wno-psabi ${MKL_LD} ${TBB_LD}")
endif()

# cscv lib compiled from */*.cc
file(GLOB_RECURSE CSCV_SRC "*/*.cc")
add_library(cscv STATIC ${CSCV_SRC})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# cscv test
# add_subdirectory(unit-test)

add_executable(cscv_test_performer unit-test/04-cscv-exp/06-test-performer/u04_06.cc)

target_link_libraries(cscv_test_performer cscv)
