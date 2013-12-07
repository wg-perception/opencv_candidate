add_library(opencv_candidate datamatrix.cpp
                             lsh.cpp
                             feature2d.cpp
)

target_link_libraries(opencv_candidate
  ${OpenCV_LIBS}
  )
set_target_properties(opencv_candidate PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")

#set_target_properties(opencv_candidate PROPERTIES COMPILE_FLAGS "-msse4.2")

install(TARGETS opencv_candidate
    DESTINATION lib
    )
#set_target_properties(opencv_candidate PROPERTIES
#  VERSION ${object_recognition_VERSION}
#  SOVERSION ${object_recognition_SOVERSION}
#  )

# create some Python wrappers
set(Python_ADDITIONAL_VERSIONS 2.7)
find_package(PythonLibs REQUIRED)

include_directories(SYSTEM ${PYTHON_INCLUDE_PATH})

add_library(pyopencv_candidate pyopencv_candidate.cpp)
target_link_libraries(pyopencv_candidate)

set_target_properties(pyopencv_candidate PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}/${PROJECT_NAME}/boost/
                      PREFIX ""
)
