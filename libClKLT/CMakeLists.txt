# Define the project's name
project(libClKLT)

cmake_minimum_required(VERSION 2.6)

# Add all the files we're interested in (mostly for QTCreator)
file(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB SRC_CL ${PROJECT_SOURCE_DIR}/src/*.cl)
file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/headers/*.h)

# Define the included directories
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/headers)

# Try to find the needed packages
# Nothing yet

# Set debug or release configurations
if( CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]" )
    message( "INFO: DEBUG BUILD" )
    # for debug type builds, turn on verbose makefiles
    SET(CMAKE_VERBOSE_MAKEFILE ON)

    # Tell other CMake files that we're doing a debug build
    SET( DEBUG_BUILD 1 )

    # Tell C/C++ that we're doing a debug build
    ADD_DEFINITIONS( -DDEBUG )
endif()

if( CMAKE_BUILD_TYPE MATCHES "[Rr][Ee][Ll][Ee][Aa][Ss][Ee]" )
    message( "INFO: RELEASE BUILD" )
endif()

# Create static and dynamic libraries
add_library (ClKLT_static STATIC ${SRC} ${HEADERS})
add_library (ClKLT_dynamic SHARED ${SRC} ${HEADERS})

# Set the link libraries :
TARGET_LINK_LIBRARIES(ClKLT_static ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(ClKLT_dynamic ${Boost_LIBRARIES} )

### Define the installation procedure
install(TARGETS ClKLT_dynamic ClKLT_static DESTINATION ${PROJECT_SOURCE_DIR}/lib)
install(FILES ${HEADERS} DESTINATION ${PROJECT_SOURCE_DIR}/include)
