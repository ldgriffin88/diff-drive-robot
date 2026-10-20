import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ubuntu/EN613/final/final_template_jazzy/src/install/gazebo_controller'
