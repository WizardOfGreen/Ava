cmake_minimum_required(VERSION 3.20)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded&<&CONFIG:Debug>:Debug>")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded&<&CONFIG:Release>:Release>")

project(main)
   
include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" )


add_executable(main ${SRC_FILES})
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

add_executable(main_test tests/test.cpp)
add_test(NAME MainTests COMMAND main_test)
target_include_directories(main_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)

add_executable(math_test tests/math_test.cpp)
target_include_directories(math_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_test(NAME MathTests COMMAND math_test)