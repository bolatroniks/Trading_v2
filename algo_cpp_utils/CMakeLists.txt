cmake_minimum_required(VERSION 2.6)
project(algo_cpp_utils)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")

set (CMAKE_EXPORT_COMPILE COMMANDS ON)
#add_subdirectory(build)

set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS
                system
                thread
             REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})



add_library(algo_cpp_utils SHARED cpp_utils.cpp)

target_link_libraries(algo_cpp_utils
        boost_python
        python2.7
    )

#install(TARGETS algo_cpp_utils RUNTIME DESTINATION bin)
