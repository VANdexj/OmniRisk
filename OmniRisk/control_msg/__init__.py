try:
    # Prefer the ROS-installed package (stays in sync with .msg files automatically)
    from quadrotor_msgs.msg import PositionCommand
except ImportError:
    # Fallback to local copy generated from Controller/src/utils/quadrotor_msgs/msg/PositionCommand.msg
    # Regenerate with: see OmniRisk/control_msg/README_regen.md
    from ._PositionCommand import PositionCommand
