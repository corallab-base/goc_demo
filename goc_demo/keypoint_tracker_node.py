#!/usr/bin/env python3
import os
import sys
import time
import copy
from importlib.resources import files
from collections import deque

import numpy as np
import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32

import torch
import sam2
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def imgmsg_to_numpy(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        ch, dtype = 3, np.uint8
    elif enc in ("mono8",):
        ch, dtype = 1, np.uint8
    elif enc in ("16uc1", "mono16"):
        ch, dtype = 1, np.uint16
    else:
        raise NotImplementedError(f"Unsupported encoding: {msg.encoding}")

    # Fast view into the ROS data buffer; respects row stride (step)
    row_stride = msg.step // np.dtype(dtype).itemsize
    arr = np.frombuffer(msg.data, dtype=dtype)
    if ch == 1:
        img = arr.reshape((msg.height, row_stride))[:, :msg.width]
    else:
        img = arr.reshape((msg.height, row_stride // ch, ch))[:, :msg.width, :]

    # (Optional) copy to make it writable: img = img.copy()
    return img


def numpy_to_imgmsg(img: np.ndarray, frame_id: str, encoding="bgr8") -> Image:
    msg = Image()
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    step_channels = (img.shape[2] if img.ndim == 3 else 1)
    msg.step = msg.width * step_channels * img.dtype.itemsize
    msg.data = memoryview(img.tobytes())
    return msg


# =================== Config (env overrides ok) ===================
SAM2_CFG   = os.environ.get("SAM2_CFG", "sam2.1_hiera_s.yaml")
SAM2_CKPT  = str(files(sam2) / "checkpoints" / "sam2.1_hiera_small.pt")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE        = 60     # sliding buffer (frames) if you switch to windowed API later
FRAME_CACHE_LIMIT  = 18     # when using add_new_frame API, rollover & reseed after this many frames
MASK_LOGIT_THR     = 0.0
SHOW_ANNOTATED     = True   # set False for headless
WINDOW_NAME        = "SAM2 Click Tracker (click to add object | q=quit)"

# ================================================================


def mask_centroid_bool(m: np.ndarray):
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


class SAM2ClickTracker:
    """
    Minimal click-to-track with SAM 2 using the 'video predictor' incremental API.

    - add_point(image_rgb, x, y) seeds a new object mask via SAM2ImagePredictor points.
    - add_frame(image_rgb) appends a frame and propagates masks to it.
    - publish centroids: read from latest_masks_by_id (per-object bool mask).
    - To bound memory, when the internal cached image tensor grows past FRAME_CACHE_LIMIT,
      we "rollover + reseed" (re-init state, add current frame once, re-add masks for all objects).
    """

    def __init__(self, device=DEVICE, cfg=SAM2_CFG, ckpt=SAM2_CKPT):
        self.device = device
        # image model for point prompting
        sam_model = build_sam2(cfg, ckpt, device=device)
        self.img_predictor = SAM2ImagePredictor(sam_model)

        # video predictor (incremental)
        self.video_predictor = build_sam2_video_predictor(cfg, ckpt)
        self.state = self.video_predictor.init_state()
        self.state["images"] = torch.empty((0, 3, 1024, 1024), device=device)
        self.state["video_height"] = None
        self.state["video_width"] = None

        # tracking storage
        self.objects_count = 0
        self.latest_masks_by_id = {}  # obj_id -> bool HxW
        self.last_frame_idx = -1

    def _ensure_size(self, image_rgb: np.ndarray):
        if self.state["video_height"] is None:
            H, W = image_rgb.shape[:2]
            self.state["video_height"] = H
            self.state["video_width"] = W

    def add_point(self, image_rgb: np.ndarray, x: int, y: int):
        """Add a positive point → seed a new object on this frame."""
        self._ensure_size(image_rgb)
        self.img_predictor.set_image(image_rgb)

        pt = np.array([[x, y]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)
        masks, scores, logits = self.img_predictor.predict(
            point_coords=pt[None, ...],
            point_labels=lbl[None, ...],
            box=None,
            multimask_output=False,
        )
        m = (masks[0] > 0).astype(np.uint8)
        if m.sum() == 0:
            print("[sam2_tracker] Click produced empty mask; ignoring.")
            return

        self.objects_count += 1
        oid = self.objects_count
        m_bool = m.astype(bool)
        self.latest_masks_by_id[oid] = m_bool

        # Add current frame and seed
        frame_idx = self.video_predictor.add_new_frame(self.state, image_rgb)
        self.video_predictor.reset_state(self.state)
        _frame_idx, _, _ = self.video_predictor.add_new_mask(self.state, frame_idx, oid, m_bool)
        self.last_frame_idx = frame_idx

        print(f"[sam2_tracker] Added object id={oid} at ({x},{y})")

    def _rollover_reseed(self, image_rgb: np.ndarray) -> int:
        """Bound memory: rebuild state, add current frame once, reseed all objects with latest masks."""
        seeds = {int(oid): m for oid, m in self.latest_masks_by_id.items() if m is not None}

        self.state = self.video_predictor.init_state()
        self.state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
        H, W = image_rgb.shape[:2]
        self.state["video_height"] = H
        self.state["video_width"] = W

        frame_idx = self.video_predictor.add_new_frame(self.state, image_rgb)
        self.video_predictor.reset_state(self.state)
        for oid, m in seeds.items():
            _f, _, _ = self.video_predictor.add_new_mask(self.state, frame_idx, oid, m.astype(bool))
        self.last_frame_idx = frame_idx
        return frame_idx

    def add_frame(self, image_rgb: np.ndarray):
        """Append new frame and propagate all tracked masks onto it. Returns (frame_idx, latest_masks_by_id)."""
        if len(self.latest_masks_by_id) == 0:
            # nothing tracked yet
            return None, self.latest_masks_by_id

        # bound memory
        if self.state["images"].shape[0] > FRAME_CACHE_LIMIT:
            frame_idx = self._rollover_reseed(image_rgb)
            rolled = True
        else:
            frame_idx = self.video_predictor.add_new_frame(self.state, image_rgb)
            rolled = False

        # propagate
        frame_idx, obj_ids, video_res_masks = self.video_predictor.infer_single_frame(
            inference_state=self.state, frame_idx=frame_idx
        )

        # refresh latest per-object masks
        for i, oid in enumerate(obj_ids):
            m_bool = (video_res_masks[i] > MASK_LOGIT_THR)[0].detach().cpu().numpy().astype(bool)
            self.latest_masks_by_id[int(oid)] = m_bool

        self.last_frame_idx = frame_idx
        return frame_idx, self.latest_masks_by_id


class SAM2ClickTrackerNode(Node):
    """
    ROS 2 node:
      - Subscribes to color images (e.g., /camera/color/image_raw)
      - OpenCV window for click-to-add points
      - Publishes centroids of tracked objects (pixel coordinates) on ~/centroids_px (sensor_msgs/PointCloud)
      - Optionally publishes annotated image on ~/annotated (sensor_msgs/Image)
    """

    def __init__(self):
        super().__init__("sam2_click_tracker_node")
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("publish_annotated", True)

        color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        self.publish_annotated = self.get_parameter("publish_annotated").get_parameter_value().bool_value

        self.sub = self.create_subscription(Image, color_topic, self.color_cb, 10)

        self.centroids_pub = self.create_publisher(PointCloud, "~/centroids_px", 10)
        self.annotated_pub = self.create_publisher(Image, "~/annotated", 10) if self.publish_annotated else None

        self.tracker = SAM2ClickTracker()

        # OpenCV window + click handling (optional headless)
        self.pending_clicks = []
        if SHOW_ANNOTATED:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            def on_mouse(event, x, y, flags, userdata):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.pending_clicks.append((x, y))
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        # small timer to service OpenCV events
        self.ui_timer = self.create_timer(0.001, self._ui_spin)

        self.get_logger().info(f"Subscribed to: {color_topic}")
        self.get_logger().info("Publishing centroids on: ~/centroids_px (sensor_msgs/PointCloud)")

    def _ui_spin(self):
        if SHOW_ANNOTATED:
            # let OpenCV handle UI events
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Quit signal (q) received; shutting down.")
                rclpy.shutdown()

    def color_cb(self, msg: Image):
        # Convert ROS Image -> BGR, then → RGB for models
        frame_rgb = imgmsg_to_numpy(msg)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Handle any pending clicks on THIS frame
        if self.pending_clicks:
            for (x, y) in self.pending_clicks:
                self.tracker.add_point(frame_rgb, x, y)
            self.pending_clicks.clear()

        # Run tracker
        frame_idx, masks_by_id = self.tracker.add_frame(frame_rgb)

        # Build & publish centroids (pixel coordinates)
        cloud = PointCloud()
        cloud.header = msg.header  # keep same time/frame_id
        ids_channel = ChannelFloat32()
        ids_channel.name = "id"

        if masks_by_id:
            H, W = frame_rgb.shape[:2]
            for oid, m in masks_by_id.items():
                c = mask_centroid_bool(m)
                if c is None:
                    continue
                # NOTE: publishing pixel coords as x,y (z=0). If you'd rather normalize, divide by W/H.
                cloud.points.append(Point32(x=float(c[0]), y=float(c[1]), z=0.0))
                ids_channel.values.append(float(oid))

        cloud.channels.append(ids_channel)
        self.centroids_pub.publish(cloud)

        # Optional annotated view/publish
        if SHOW_ANNOTATED or self.publish_annotated:
            vis = frame_bgr.copy()
            for oid, m in masks_by_id.items():
                # mask outline
                m_u8 = (m.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
                c = mask_centroid_bool(m)
                if c is not None:
                    cv2.circle(vis, (int(c[0]), int(c[1])), 4, (0, 0, 255), -1)
                    cv2.putText(vis, f"id{oid}", (int(c[0]) + 6, int(c[1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            if SHOW_ANNOTATED:
                cv2.imshow(WINDOW_NAME, vis)
            if self.publish_annotated and self.annotated_pub is not None:
                self.annotated_pub.publish(numpy_to_imgmsg(vis, "0", encoding="bgr8"))


def main():
    rclpy.init()
    node = SAM2ClickTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
