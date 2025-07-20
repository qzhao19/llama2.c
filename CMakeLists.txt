cmake_minimum_required(VERSION 3.22.1)
project(llama2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(USE_OPENMP "Use OpenMP for parallelism" OFF)    # Default is not using OpenMP
option(USE_SSE "Use OpenMP for parallelism" ON)        # Default is not using OpenMP
option(USE_AVX "Use OpenMP for parallelism" ON)        # Default is not using OpenMP
option(ENABLE_INSTALL "Enable installation rules" OFF)

# set common compile flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if(USE_SSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
    add_definitions(-DUSE_SSE)
endif()

if(USE_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    add_definitions(-DUSE_AVX2)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core-avx2")

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DUSE_OPENMP)
endif()

# common source file
set(COMMON_SOURCES
    utility.cpp
)

add_executable(run ${COMMON_SOURCES} run.cpp)
target_include_directories(run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(runq ${COMMON_SOURCES} runq.cpp)
target_include_directories(runq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(ENABLE_INSTALL)
    install(TARGETS run runq
        RUNTIME DESTINATION bin
    )
endif()
