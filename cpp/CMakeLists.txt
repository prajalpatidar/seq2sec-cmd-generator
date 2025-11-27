cmake_minimum_required(VERSION 3.15)
project(seq2sec_cmd_generator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime
# Option 1: Use system installation
# find_package(onnxruntime REQUIRED)

# Option 2: Specify ONNX Runtime path manually
set(ONNXRUNTIME_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime" CACHE PATH "ONNX Runtime root directory")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_PATH}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/tokenizer.cpp
    src/seq2seq_inference.cpp
    src/main.cpp
)

# Create executable
add_executable(cmd_generator ${SOURCES})

# Link libraries
target_link_libraries(cmd_generator 
    ${ONNXRUNTIME_LIB}
)

# Install targets
install(TARGETS cmd_generator DESTINATION bin)

# Optional: Build shared library for integration
add_library(seq2sec_lib SHARED
    src/tokenizer.cpp
    src/seq2seq_inference.cpp
)

target_link_libraries(seq2sec_lib ${ONNXRUNTIME_LIB})
install(TARGETS seq2sec_lib DESTINATION lib)
install(FILES 
    include/tokenizer.h 
    include/seq2seq_inference.h 
    DESTINATION include
)
