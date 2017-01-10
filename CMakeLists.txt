cmake_minimum_required(VERSION 2.8.7)
project(photo_mosaic)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(CMAKE_PREFIX_PATH "/usr/local")
set(CMAKE_INCLUDE_PATH "${CMAKE_PREFIX_PATH}/include")
set(CMAKE_LIBRARY_PATH "${CMAKE_PREFIX_PATH}/lib")

# OpenCV
find_package(OpenCV 2.4.12 REQUIRED core highgui imgproc)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIBS}")
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

# Boost
find_package(Boost 1.48.0 COMPONENTS system filesystem program_options REQUIRED)
if(Boost_FOUND)
    message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
    message(FATAL_ERROR "Boost not found")
endif()
include_directories(${Boost_INCLUDE_DIRS})

# Add executable file
add_executable(demo src/demo.cc src/mosaic.cc)
target_link_libraries(demo ${OpenCV_LIBS} ${Boost_LIBRARIES})
