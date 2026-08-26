cmake_minimum_required(VERSION 3.20)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded&<&CONFIG:Debug>:Debug>")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded&<&CONFIG:Release>:Release>")

project(main)
   
include(CTest)
enable_testing()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-g -O0 --coverage -fprofile-arcs -ftest-coverage)
    add_link_options(--coverage)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" )

add_executable(main ${SRC_FILES})
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

add_executable(act_sigmoid_test tests/act_sigmoid_test.cpp)
target_include_directories(act_sigmoid_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_test(NAME Act_SigmoidTest COMMAND act_sigmoid_test)

add_executable(act_relu_test tests/act_relu_test.cpp)
target_include_directories(act_relu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_test(NAME Act_ReluTest COMMAND act_relu_test)

add_executable(act_tanh_test tests/act_tanh_test.cpp)
target_include_directories(act_tanh_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_test(NAME Act_TanHTest COMMAND act_tanh_test)

add_executable(act_softplus tests/act_softplus.cpp)
target_include_directories(act_softplus PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_test(NAME Act_SoftPlus COMMAND act_softplus)

add_executable(act_stepfunc tests/act_stepfunc.cpp)
target_include_directories(act_stepfunc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_test(NAME Act_StepFunc COMMAND act_stepfunc)

add_executable(loss_mse tests/loss_mse.cpp)
target_include_directories(loss_mse PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_test(NAME Loss_MSE COMMAND loss_mse)

add_executable(loss_rootmse tests/loss_rootmse.cpp)
target_include_directories(loss_rootmse PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_test(NAME Loss_RootMSE COMMAND loss_rootmse)

add_executable(loss_mae tests/loss_mae.cpp)
target_include_directories(loss_mae PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_test(NAME Loss_MAE COMMAND loss_mae)

add_executable(perc_test tests/perc_test.cpp src/Perceptron.cpp) 
target_include_directories(perc_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_test(NAME Perc_Test COMMAND perc_test)