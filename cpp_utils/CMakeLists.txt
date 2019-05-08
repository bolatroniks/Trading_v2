cmake_minimum_required(VERSION 3.5)
project(cpp_utils)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set (CMAKE_CXX FLAGS "${CMAKE_CXX_FLAGS} /MP")

set ( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set (BOOST_ROOT_DIR "/home/joanna/Downloads/boost_1_58_0" CACHE STRING "Set this...")
set (BOOST_ROOT $(BOOST_ROOT_DIR))

message ("BOOST_ROOT: " $(BOOST_ROOT))
