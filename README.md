Commands to run Driver and Detection Overlay

'''bash
ros2 launch depthai_ros_driver_v3 driver.launch.py \
  params_file:=/absolute/path/to/params.yaml
'''

'''bash
ros2 component load /oak_container depthai_filters_v3 depthai_filters::Detection2DOverlay
'''
