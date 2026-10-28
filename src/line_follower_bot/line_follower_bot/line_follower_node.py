"""Camera-based line follower with confidence-aware PID control."""

from collections import deque
from typing import Optional

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from .image_process import detect_line, make_debug_image


class LineFollower(Node):
    """Follow a dark track while exposing error and visual diagnostics."""

    def __init__(self) -> None:
        super().__init__("line_follower")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("image_topic", "/camera/image_raw"),
                ("cmd_vel_topic", "/cmd_vel"),
                ("debug_image_topic", "/line_follower/debug_image"),
                ("base_speed", 0.42),
                ("min_speed", 0.10),
                ("max_angular_speed", 3.5),
                ("kp", 2.8),
                ("ki", 0.03),
                ("kd", 0.24),
                ("integral_limit", 0.7),
                ("derivative_alpha", 0.72),
                ("roi_top_ratio", 0.42),
                ("threshold", 105),
                ("lost_timeout", 0.65),
                ("recovery_speed", 0.16),
            ],
        )
        get = lambda name: self.get_parameter(name).value
        self.base_speed = float(get("base_speed"))
        self.min_speed = float(get("min_speed"))
        self.max_angular_speed = float(get("max_angular_speed"))
        self.kp = float(get("kp"))
        self.ki = float(get("ki"))
        self.kd = float(get("kd"))
        self.integral_limit = float(get("integral_limit"))
        self.derivative_alpha = float(get("derivative_alpha"))
        self.roi_top_ratio = float(get("roi_top_ratio"))
        self.threshold = int(get("threshold"))
        self.lost_timeout = float(get("lost_timeout"))
        self.recovery_speed = float(get("recovery_speed"))

        self.bridge = CvBridge()
        self.previous_x: Optional[float] = None
        self.previous_error = 0.0
        self.filtered_derivative = 0.0
        self.integral = 0.0
        self.last_seen = self.get_clock().now()
        self.last_direction = 1.0
        self.error_history = deque(maxlen=5)

        self.cmd_pub = self.create_publisher(Twist, str(get("cmd_vel_topic")), 10)
        self.debug_pub = self.create_publisher(Image, str(get("debug_image_topic")), 10)
        self.error_pub = self.create_publisher(Float32, "/line_follower/error", 10)
        self.confidence_pub = self.create_publisher(Float32, "/line_follower/confidence", 10)
        self.state_pub = self.create_publisher(String, "/line_follower/state", 10)
        self.image_sub = self.create_subscription(
            Image, str(get("image_topic")), self.image_callback, 10
        )
        self.watchdog = self.create_timer(0.15, self.watchdog_callback)

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def watchdog_callback(self) -> None:
        if (self.get_clock().now() - self.last_seen).nanoseconds / 1e9 > self.lost_timeout:
            self.publish_stop()
            state = String()
            state.data = "waiting_for_camera"
            self.state_pub.publish(state)

    def image_callback(self, message: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except (ValueError, cv2.error) as error:
            self.get_logger().error(f"Could not convert camera frame: {error}")
            return

        observation = detect_line(
            frame,
            self.previous_x,
            roi_top_ratio=self.roi_top_ratio,
            threshold=self.threshold,
        )
        now = self.get_clock().now()
        dt = max(0.001, (now - self.last_seen).nanoseconds / 1e9)
        self.last_seen = now
        if observation.line_x is not None:
            self.previous_x = observation.line_x
            self.last_direction = 1.0 if observation.error >= 0.0 else -1.0
            self.error_history.append(observation.error)
            self.integral = max(
                -self.integral_limit,
                min(self.integral_limit, self.integral + observation.error * dt),
            )
            derivative = (observation.error - self.previous_error) / dt
            self.filtered_derivative = (
                self.derivative_alpha * self.filtered_derivative
                + (1.0 - self.derivative_alpha) * derivative
            )
            self.previous_error = observation.error
            angular = (
                self.kp * observation.error
                + self.ki * self.integral
                + self.kd * self.filtered_derivative
            )
            angular = max(-self.max_angular_speed, min(self.max_angular_speed, angular))
            confidence = max(0.15, observation.confidence)
            speed = self.base_speed * (1.0 - 0.72 * min(1.0, abs(observation.error)))
            speed *= 0.65 + 0.35 * confidence
            speed = max(self.min_speed, speed)
            state_text = "tracking"
        else:
            self.integral *= 0.8
            angular = self.last_direction * self.max_angular_speed * 0.58
            speed = self.recovery_speed
            state_text = "recovering"

        command = Twist()
        command.linear.x = float(speed)
        command.angular.z = float(angular)
        self.cmd_pub.publish(command)

        error_message = Float32()
        error_message.data = float(observation.error)
        self.error_pub.publish(error_message)
        confidence_message = Float32()
        confidence_message.data = float(observation.confidence)
        self.confidence_pub.publish(confidence_message)
        state = String()
        state.data = state_text
        self.state_pub.publish(state)

        try:
            debug = make_debug_image(frame, observation, observation.error, angular)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))
        except (ValueError, cv2.error) as error:
            self.get_logger().error(f"Could not publish debug frame: {error}")


def main(args=None) -> None:
    """Run the line follower node."""
    import rclpy

    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()
