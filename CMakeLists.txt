# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take their
# definition
# and dump it in the cache along with proper documentation, otherwise set
# CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
#
#
#IF(DEFINED CMAKE_BUILD_TYPE)
#   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
#build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
#RelWithDebInfo MinSizeRel.")
#ELSE()
#   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build,
#   options
#are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo
#MinSizeRel.")
#ENDIF()

#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -std=c++0x" CACHE STRING "Debug options."
#FORCE)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -std=c++0x" CACHE STRING "Debug options."
#FORCE)
#set(DCMAKE_CXX_COMPILER=/cvmfs/lhcb.cern.ch/lib/lcg/releases/gcc/4.9.3/x86_64-slc6/bin/g++)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{DOOFITSYS}/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{DOOCORESYS}/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{DOOSELECTIONSYS}/cmake/Modules/")


set(BASEPATH "${CMAKE_SOURCE_DIR}/src")
include_directories(${BASEPATH})

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

# Project Definition
project(Bd2JpsiKS)
cmake_minimum_required(VERSION 2.6)

if(${APPLE})
  MESSAGE( STATUS "Building for Mac OS X, switching on C++11 flags for Mac OSX/clang" )
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++")
endif(${APPLE})
IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  MESSAGE( STATUS "Building for Linux, switching off C++11 flags for Linux/gcc" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
set(BOOST_ROOT "/cvmfs/lhcb.cern.ch/lib/lcg/releases/LCG_87/Boost/1.62.0/x86_64-slc6-gcc49-opt")
FIND_PACKAGE(Boost 1.56 COMPONENTS program_options filesystem thread system regex random REQUIRED)
find_package(ROOT REQUIRED)
find_package(DooCore REQUIRED)
find_package(DooSelection REQUIRED)
find_package(DooFit REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

include_directories(${DooCore_INCLUDE_DIR})
link_directories(${DooCore_LIBRARY_DIR})

include_directories(${DooSelection_INCLUDE_DIR})
link_directories(${DooSelection_LIBRARY_DIR})

include_directories(${DooFit_INCLUDE_DIR})
link_directories(${DooFit_LIBRARY_DIR})

include_directories(${ROOT_INCLUDE_DIR})

set(ALL_LIBRARIES ${DooFit_LIBRARIES} ${DooSelection_LIBRARIES} ${DooCore_LIBRARIES} ${ROOT_LIBRARIES} ${ROOFIT_LIBRARIES} ${Boost_LIBRARIES} ${ADDITIONAL_LIBRARIES} -lTMVA "-lTreePlayer")

add_subdirectory(main)
