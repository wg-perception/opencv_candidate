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

# add the model capture app/library
find_package(PCL REQUIRED COMPONENTS io filters visualization features segmentation surface)
include_directories(SYSTEM ${PCL_INCLUDE_DIRS}
                           /usr/include/suitesparse
)

# Find Eigen3
FIND_PACKAGE(Eigen)
find_package(libg2o)

include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})

add_executable(model_capture samples/reconst3d/reconst3d.hpp
                             samples/reconst3d/reconst3d_init.cpp

                             samples/reconst3d/graph_optimizations.hpp
                             samples/reconst3d/ocv_pcl_convert.hpp

                             samples/reconst3d/capture.cpp
                             samples/reconst3d/reconstruction.cpp
                             samples/reconst3d/model.cpp

                             samples/reconst3d/load_data.cpp
                             samples/reconst3d/mask_data.cpp

                             samples/reconst3d/graph_se3.cpp
                             samples/reconst3d/graph_se3_rgbd_icp.cpp
                             samples/reconst3d/graph_se3_rgbd_icp_model.cpp

                             samples/model_capture.cpp
              )
target_link_libraries(model_capture ${catkin_LIBRARIES}
                                    ${OpenCV_LIBRARIES}
                                    opencv_rgbd
                                    ${libg2o_LIBRARIES}
                                    ${PCL_LIBRARIES}
                                    cholmod
                                    cxsparse
)

set_target_properties(model_capture PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
