# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
HumanWholeBodyPredictor - A reusable, thread-safe predictor for SAM 3D Body.

This class provides a clean interface for processing images/videos with SAM 3D Body,
similar to the HamerPredictor pattern. It handles:
- Single images or image sequences (videos)
- Input format: [T, H, W, 3] numpy arrays (RGB, 0-255)
- Returns: results dict and rendered visualizations
- Thread-safe for concurrent use
"""

import threading
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Import SAM 3D Body components
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample_together, visualize_overlay
from tools.build_detector import HumanDetector
from tools.build_fov_estimator import FOVEstimator


class HumanWholeBodyPredictor:
    """
    A reusable, thread-safe predictor for whole-body human mesh recovery using SAM 3D Body.

    This class can process single images or sequences of images (videos) and returns
    both the raw prediction results and rendered visualizations.

    Example:
        >>> predictor = HumanWholeBodyPredictor(
        ...     checkpoint_path="./checkpoints/sam-3d-body-dinov3/model.ckpt",
        ...     mhr_path="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
        ... )
        >>>
        >>> # Process single image
        >>> img_rgb = cv2.imread("image.jpg")[:, :, ::-1]  # BGR -> RGB
        >>> results = predictor.predict(img_rgb)
        >>>
        >>> # Process video sequence
        >>> video_frames = np.stack([frame1, frame2, ...])  # (T, H, W, 3)
        >>> results = predictor.predict(video_frames, verbose=True)
    """

    def __init__(
        self,
        checkpoint_path: str,
        mhr_path: str,
        detector_name: str = "vitdet",
        detector_path: Optional[str] = None,
        fov_name: str = "moge2",
        fov_path: Optional[str] = None,
        bbox_thresh: float = 0.8,
        use_mask: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the HumanWholeBodyPredictor.

        Args:
            checkpoint_path: Path to SAM 3D Body model checkpoint (.ckpt file)
            mhr_path: Path to MoHR assets folder (mhr_model.pt)
            detector_name: Human detection model name ('vitdet' or other supported)
            detector_path: Optional path to detector model folder
            fov_name: FOV estimation model name ('moge2' or other supported)
            fov_path: Optional path to FOV estimator model folder
            bbox_thresh: Bounding box detection threshold (default: 0.8)
            use_mask: Whether to use mask-conditioned prediction (default: False)
            device: Torch device (default: auto-detect CUDA)
        """
        self.checkpoint_path = checkpoint_path
        self.mhr_path = mhr_path
        self.detector_name = detector_name
        self.detector_path = detector_path or ""
        self.fov_name = fov_name
        self.fov_path = fov_path or ""
        self.bbox_thresh = bbox_thresh
        self.use_mask = use_mask

        # Thread safety lock
        self._lock = threading.Lock()

        # Device setup
        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device

        print(f"Loading SAM 3D Body models on {self.device}...")

        # Load SAM 3D Body model
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            mhr_path=self.mhr_path,
        )

        # Initialize optional components
        human_detector = None
        if self.detector_name:
            print(f"Loading {self.detector_name} detector...")
            human_detector = HumanDetector(
                name=self.detector_name,
                device=self.device,
                path=self.detector_path,
            )

        fov_estimator = None
        if self.fov_name:
            print(f"Loading {self.fov_name} FOV estimator...")
            fov_estimator = FOVEstimator(
                name=self.fov_name,
                device=self.device,
                path=self.fov_path,
            )

        # Create estimator
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.model_cfg,
            human_detector=human_detector,
            human_segmentor=None,  # Can be added if needed
            fov_estimator=fov_estimator,
        )

        print("Initialization complete.")

    def predict(
        self,
        rgb_input: Union[np.ndarray, List[np.ndarray]],
        full_frame: bool = True,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Process input RGB image(s) and return predictions with visualizations.

        Args:
            rgb_input: numpy array of shape (H, W, 3) or (T, H, W, 3), or list of arrays.
                       Values should be 0-255 (uint8), RGB format.
            full_frame: Boolean, whether to return full frame visualization (default: True)
            verbose: Boolean, print progress (default: False)

        Returns:
            results: list of dicts (one per frame). Each dict contains:
                - 'frame_idx': index of the frame
                - 'outputs': list of person predictions (from SAM3DBodyEstimator.process_one_image)
                    Each person dict contains:
                    - 'bbox': bounding box [x1, y1, x2, y2]
                    - 'pred_vertices': 3D mesh vertices (N, 3)
                    - 'pred_keypoints_3d': 3D keypoints
                    - 'pred_keypoints_2d': 2D keypoints
                    - 'pred_cam_t': camera translation
                    - 'focal_length': focal length
                    - ... (other MHR parameters)
                - 'visualization': np.ndarray (H, W, 3) BGR image (if full_frame=True)
                    Shows original image + keypoints + mesh overlay + side view
                    Format is BGR (uint8) suitable for cv2.imwrite()
        """
        # Thread-safe execution
        with self._lock:
            return self._predict_impl(rgb_input, full_frame, verbose)

    def _predict_impl(
        self,
        rgb_input: Union[np.ndarray, List[np.ndarray]],
        full_frame: bool,
        verbose: bool,
    ) -> List[Dict]:
        """Internal implementation (called within lock)."""
        # Normalize input to list of frames or (T, H, W, 3)
        frames = None
        if isinstance(rgb_input, np.ndarray):
            if len(rgb_input.shape) == 3:
                frames = rgb_input[None, ...]  # Add T dim: (H, W, 3) -> (1, H, W, 3)
            elif len(rgb_input.shape) == 4:
                frames = rgb_input  # Already (T, H, W, 3)
            else:
                raise ValueError(
                    f"Input shape {rgb_input.shape} not supported. "
                    f"Expected (H, W, 3) or (T, H, W, 3)"
                )
        elif isinstance(rgb_input, list):
            frames = np.stack(rgb_input)  # List of arrays -> (T, H, W, 3)
        else:
            raise ValueError("Input must be numpy array or list of numpy arrays.")

        # Ensure uint8 and RGB
        if frames.dtype != np.uint8:
            if frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            else:
                frames = frames.astype(np.uint8)

        results = []

        # Process frames
        frames_iter = (
            tqdm(enumerate(frames), total=len(frames), desc="Processing frames")
            if verbose
            else enumerate(frames)
        )

        for frame_idx, img_rgb in frames_iter:
            # Process image with SAM 3D Body
            # process_one_image expects RGB when passed as numpy array
            outputs = self.estimator.process_one_image(
                img_rgb,  # RGB format
                bbox_thr=self.bbox_thresh,
                use_mask=self.use_mask,
            )

            frame_result = {
                "frame_idx": frame_idx,
                "outputs": outputs,
                "visualization": None,
            }

            # Generate visualization if requested
            if full_frame:
                if len(outputs) > 0:
                    # visualize_overlay expects BGR input and returns BGR
                    img_bgr = img_rgb[:, :, ::-1].copy()
                    rend_img = visualize_overlay(img_bgr, outputs, self.estimator.faces)
                    # rend_img is already BGR (uint8), suitable for cv2.imwrite
                    # Store as BGR for consistency with OpenCV operations
                    frame_result["visualization"] = rend_img.astype(np.uint8)
                else:
                    # No detections, return original image (convert RGB to BGR for consistency)
                    frame_result["visualization"] = img_rgb[:, :, ::-1].copy()

            results.append(frame_result)

        return results

    def predict_from_path(
        self,
        input_path: str,
        full_frame: bool = True,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Process input from file path (image or video).

        Args:
            input_path: Path to image file (.jpg, .png, etc.) or video file (.mp4, .avi, etc.)
            full_frame: Boolean, whether to return full frame visualization
            verbose: Boolean, print progress

        Returns:
            results: Same format as predict() method
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Check if it's a video or image
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        if input_path.suffix.lower() in video_extensions:
            # Load video
            cap = cv2.VideoCapture(str(input_path))
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # cv2 reads BGR, convert to RGB
                frames.append(frame[:, :, ::-1])

            cap.release()

            if len(frames) == 0:
                raise ValueError(f"No frames extracted from video: {input_path}")

            if verbose:
                print(f"Loaded {len(frames)} frames from video (FPS: {fps:.2f})")

            results = self.predict(
                np.stack(frames), full_frame=full_frame, verbose=verbose
            )

            # Add metadata
            for res in results:
                res["fps"] = fps
                res["video_path"] = str(input_path)

            return results

        elif input_path.suffix.lower() in image_extensions:
            # Load single image
            img = cv2.imread(str(input_path))
            if img is None:
                raise ValueError(f"Failed to load image: {input_path}")

            img_rgb = img[:, :, ::-1]  # BGR -> RGB
            results = self.predict(img_rgb, full_frame=full_frame, verbose=verbose)

            # Add metadata
            results[0]["image_path"] = str(input_path)

            return results

        else:
            raise ValueError(
                f"Unsupported file format: {input_path.suffix}. "
                f"Supported: {video_extensions | image_extensions}"
            )


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HumanWholeBodyPredictor Demo for Video/Image"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video or image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_wholebody",
        help="Path to output folder",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to SAM 3D Body checkpoint",
    )
    parser.add_argument(
        "--mhr_path",
        type=str,
        required=True,
        help="Path to MHR model file",
    )
    parser.add_argument(
        "--bbox_thresh",
        type=float,
        default=0.8,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction",
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = HumanWholeBodyPredictor(
        checkpoint_path=args.checkpoint_path,
        mhr_path=args.mhr_path,
        bbox_thresh=args.bbox_thresh,
        use_mask=args.use_mask,
    )

    # Process input
    import os
    import time

    start_time = time.time()
    results = predictor.predict_from_path(
        args.input,
        full_frame=True,
        verbose=True,
    )
    end_time = time.time()

    total_time = end_time - start_time
    num_frames = len(results)
    avg_fps = num_frames / total_time if total_time > 0 else 0

    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Processed {num_frames} frame(s)")
    print(f"Average FPS: {avg_fps:.2f}")

    # Save outputs
    os.makedirs(args.output, exist_ok=True)

    # Check if it's a video or image
    input_path = Path(args.input)
    is_video = input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if is_video:
        # Save video
        fps = results[0].get("fps", 30.0)
        if len(results) > 0 and results[0]["visualization"] is not None:
            h, w, _ = results[0]["visualization"].shape
            # Use H.264 codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_filename = os.path.join(args.output, f"out_{input_path.name}")
            out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

            for res in results:
                vis = res["visualization"]
                if vis is not None:
                    # visualization is already in BGR format (from visualize_sample_together)
                    # Ensure it's uint8
                    vis_uint8 = vis.astype(np.uint8) if vis.dtype != np.uint8 else vis
                    out.write(vis_uint8)

            out.release()
            print(f"Saved output video to {output_filename}")

    else:
        # Save single image
        if len(results) > 0 and results[0]["visualization"] is not None:
            output_filename = os.path.join(args.output, f"out_{input_path.stem}.jpg")
            vis = results[0]["visualization"]
            # Ensure uint8 format
            vis_uint8 = vis.astype(np.uint8) if vis.dtype != np.uint8 else vis
            # visualization is already in BGR format (from visualize_sample_together)
            cv2.imwrite(output_filename, vis_uint8)
            print(f"Saved output image to {output_filename}")

            # Also save raw results
            # Note: outputs contain dictionaries with varying shapes, so we use pickle
            import pickle

            results_filename = os.path.join(
                args.output, f"results_{input_path.stem}.pkl"
            )
            # Extract key data for saving
            save_data = {
                "outputs": [res["outputs"] for res in results],
                "frame_indices": [res["frame_idx"] for res in results],
            }
            with open(results_filename, "wb") as f:
                pickle.dump(save_data, f)
            print(f"Saved raw results to {results_filename}")
