cmake_minimum_required(VERSION 3.20)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded&<&CONFIG:Debug>:Debug>")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded&<&CONFIG:Release>:Release>")

project(main)
   
include(CTest)
enable_testing()

file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main ${SRC_FILES})

target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)