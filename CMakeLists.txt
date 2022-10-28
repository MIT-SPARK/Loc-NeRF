cmake_minimum_required(VERSION 2.8.3)
project(locnerf)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  sensor_msgs
  std_msgs
  rospy
)

#set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS src/nav_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
