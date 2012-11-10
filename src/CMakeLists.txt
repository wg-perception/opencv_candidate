add_subdirectory(opencv_candidate)
add_subdirectory(rgbd)

find_library(DEPTH_SENSE_LIBRARIES DepthSense /opt/DepthSenseSDK/lib)

if (NOT ${DEPTH_SENSE_LIBRARIES} STREQUAL DEPTH_SENSE_LIBRARIES-NOTFOUND)
  add_subdirectory(creative)
endif()
