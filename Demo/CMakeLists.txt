# Define the project's name
project(demoCudaLK)

cmake_minimum_required(VERSION 2.6)

# Add all the files we're interested in (mostly for QTCreator)
file(GLOB SRC ${PROJECT_SOURCE_DIR}/*.cpp)
#file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/headers/*.h)

# Set debug or release configurations
if( CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]" )
    message( "INFO: DEBUG BUILD" )
    # for debug type builds, turn on verbose makefiles
    set(CMAKE_VERBOSE_MAKEFILE ON)

    # Tell other CMake files that we're doing a debug build
    set(DEBUG_BUILD 1 )

    # Tell C/C++ that we're doing a debug build
    add_definitions( -DDEBUG )
endif()

# Get the needed lib and includes:
find_package(OpenCV REQUIRED)

find_path(CUDALK_INCLUDE_DIR PTAM/System.h
          HINTS "/usr/local/include"
          PATH_SUFFIXES PTAM)

set (CUDALK_INCLUDE_DIR ${PTAM_INCLUDE_DIR}/PTAM)

find_library(CUDALK_LIBRARY NAMES CUDALKd libCUDALKd
             HINTS "/usr/local/lib")

# Define the included directories
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OPENCV_INCLUDE_DIRS})

message("\nCudaLK lib :\n ${PTAM_LIBRARY}\n ${PTAM_INCLUDE_DIR}\n")

# Set debug or release configurations
if( CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]" )
    message( "INFO: DEBUG BUILD" )
    # for debug type builds, turn on verbose makefiles
    set(CMAKE_VERBOSE_MAKEFILE ON)

    # Tell other CMake files that we're doing a debug build
    set(DEBUG_BUILD 1 )

    # Tell C/C++ that we're doing a debug build
    add_definitions( -DDEBUG )
endif()

if( CMAKE_BUILD_TYPE MATCHES "[Rr][Ee][Ll][Ee][Aa][Ss][Ee]" )
    message( "INFO: RELEASE BUILD" )
endif()

# Set the link libraries :
set(LINK_LIBS ${OPENCV_LIBRARIES})

# Build the executable
message("Source files : ${SRC}")

add_executable(demoCudaLK ${SRC})

if( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
  set_target_properties(demoPTAM PROPERTIES COMPILE_FLAGS "-fPIC")
endif( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )

target_link_libraries(demoPTAM PTAMs  ${LINK_LIBS})

### Define the installation procedure
#install(TARGETS exe DESTINATION folder)
 
