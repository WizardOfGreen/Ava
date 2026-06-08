cmake_minimum_required(VERSION 3.20)

project(CppMatrixes)

include(CTest)
enable_testing()

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(CppMatrixes ${SRC_FILES})

target_include_directories(CppMatrixes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)