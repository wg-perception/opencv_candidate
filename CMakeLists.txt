cmake_minimum_required(VERSION 2.8)
project(opencv_candidate)

find_package(catkin REQUIRED)
find_library(DEPTH_SENSE_LIBRARIES DepthSense /opt/DepthSenseSDK/lib)

if (${DEPTH_SENSE_LIBRARIES} STREQUAL DEPTH_SENSE_LIBRARIES-NOTFOUND)
  set(OPENCV_CREATIVE "")
else()
  set(OPENCV_CREATIVE opencv_creative)
endif()

if (${catkin_VERSION} VERSION_GREATER "0.5.28")
catkin_package(DEPENDS OpenCV libg2o
               INCLUDE_DIRS include
               LIBRARIES opencv_candidate
                         ${OPENCV_CREATIVE}
                         opencv_rgbd
)
else()
catkin_stack()
catkin_project(opencv_candidate
               DEPENDS OpenCV libg2o
               INCLUDE_DIRS include
               LIBRARIES opencv_candidate
                         ${OPENCV_CREATIVE}
                         opencv_rgbd
)
endif()

add_definitions(-Wno-strict-aliasing -Wall -Werror)

find_package(OpenCV REQUIRED core imgproc highgui calib3d)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV headers: ${OpenCV_INCLUDE_DIRS}")
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
include_directories(include)

# install the include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/rgbd/include)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/opencv_candidate
        DESTINATION include
        COMPONENT main
)

add_subdirectory(src)
