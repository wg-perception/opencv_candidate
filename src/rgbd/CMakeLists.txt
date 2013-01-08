# merge some RGBD functions into a module
#set(the_description "a set of RGBD functions")
#ocv_define_module(rgbd)

find_package(OpenCV REQUIRED)

add_library(opencv_rgbd src/normal.cpp
                        src/depth_to_3d.cpp
                        src/depth_cleaner.cpp
                        src/odometry.cpp
                        src/plane.cpp
                        src/rgbd_init.cpp
                        src/utils.cpp
)

target_link_libraries(opencv_rgbd ${OpenCV_LIBS}
                                  ${OpenCV_LIBRARIES}
)

install(TARGETS opencv_rgbd
        DESTINATION lib
)

install(DIRECTORY include/opencv2/rgbd
        DESTINATION include/opencv2
        COMPONENT main
)

# Add sample
add_executable(odometry_evaluation samples/odometry_evaluation.cpp)
target_link_libraries(odometry_evaluation ${OpenCV_LIBRARIES} opencv_rgbd)

# Add some tests
add_executable(rgbd_tests test/test_main.cpp
                          test/test_normal.cpp
                          test/test_odometry.cpp
                          test/test_precomp.cpp
                          test/test_utils.cpp
)

find_library(OPENCV_TS_LIBRARY opencv_ts ${OpenCV_LIB_DIR})
target_link_libraries(rgbd_tests ${OpenCV_LIBRARIES}
                                 opencv_rgbd
                                 ${OPENCV_TS_LIBRARY}
)

add_test(rgbd_tests rgbd_tests)

# Model capture stuff
find_package(PCL REQUIRED COMPONENTS io filters visualization features segmentation surface)
include_directories(SYSTEM ${PCL_INCLUDE_DIRS}
                           /usr/include/suitesparse
)

# Find Eigen3
FIND_PACKAGE(Eigen)
find_package(libg2o)

include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})

# add model capture sample
include_directories(${reconst3d_dir}/include)
add_executable(model_capture samples/model_capture.cpp)
target_link_libraries(model_capture ${OpenCV_LIBRARIES}
                                    opencv_rgbd
                                    opencv_candidate_reconst3d
)
set_target_properties(model_capture PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
