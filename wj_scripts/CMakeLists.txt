
add_executable(test_MOT17_data test_MOT17_data.cpp)

target_link_libraries(test_MOT17_data PRIVATE motcpp)

list(APPEND OPENCV_LINK_LIBS opencv_highgui opencv_tracking) # for use of imshow, waitKey, for doing tracking

target_link_libraries(test_MOT17_data PUBLIC ${OPENCV_LINK_LIBS})