#!/usr/bin/env python3
"""
Test script for HumanWholeBodyPredictor.

This script tests the predictor with images or videos and saves rendered outputs.
Supports both single image and video processing modes.pip install pymomentum-gpu
"""

import sys
import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from human_wholebody_predictor import HumanWholeBodyPredictor


def get_checkpoint_paths():
    """Get checkpoint paths, checking if they exist."""
    checkpoint_path = (
        Path(__file__).parent / "checkpoints" / "sam-3d-body-dinov3" / "model.ckpt"
    )
    mhr_path = (
        Path(__file__).parent
        / "checkpoints"
        / "sam-3d-body-dinov3"
        / "assets"
        / "mhr_model.pt"
    )
    return checkpoint_path, mhr_path


def test_single_image():
    """Test with a single image."""
    print("Testing HumanWholeBodyPredictor with single image...")

    # Check if example image exists
    example_image = Path(__file__).parent / "example" / "human-ego.jpeg"
    if not example_image.exists():
        print(f"Example image not found: {example_image}")
        print(
            "Please provide a test image path or ensure example/human-ego.jpeg exists"
        )
        return False

    # Check if checkpoints exist
    checkpoint_path, mhr_path = get_checkpoint_paths()

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please download checkpoints first (see INSTALL.md)")
        return False

    if not mhr_path.exists():
        print(f"MHR model not found: {mhr_path}")
        print("Please download checkpoints first (see INSTALL.md)")
        return False

    try:
        # Initialize predictor
        print("\nInitializing predictor...")
        predictor = HumanWholeBodyPredictor(
            checkpoint_path=str(checkpoint_path),
            mhr_path=str(mhr_path),
            bbox_thresh=0.8,
        )

        # Load and process image
        print("\nLoading image...")
        img_bgr = cv2.imread(str(example_image))
        img_rgb = img_bgr[:, :, ::-1]  # Convert to RGB

        print("Processing image...")
        results = predictor.predict(img_rgb, full_frame=True, verbose=True)

        if len(results) == 0:
            print("ERROR: No results returned!")
            return False

        result = results[0]

        print("\nResults:")
        print(f"  Frame index: {result['frame_idx']}")
        print(f"  Number of detected persons: {len(result['outputs'])}")
        print(f"  Has visualization: {result['visualization'] is not None}")

        if result["visualization"] is not None:
            print(f"  Visualization shape: {result['visualization'].shape}")

        # Check outputs structure
        if len(result["outputs"]) > 0:
            person = result["outputs"][0]
            print(f"\nFirst person output keys: {list(person.keys())}")
            print(f"  Bbox: {person.get('bbox', 'N/A')}")
            print(
                f"  Vertices shape: {person.get('pred_vertices', np.array([])).shape}"
            )
            print(
                f"  Keypoints 3D shape: {person.get('pred_keypoints_3d', np.array([])).shape}"
            )

        print("\n✓ Test passed successfully!")

        # Save output visualization
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)

        if result["visualization"] is not None:
            # Save rendered image (BGR format for OpenCV)
            output_image_path = output_dir / "test_output.jpg"
            vis = result["visualization"]
            # Ensure uint8 format
            vis_uint8 = vis.astype(np.uint8) if vis.dtype != np.uint8 else vis
            cv2.imwrite(str(output_image_path), vis_uint8)
            print(f"\nSaved visualization to: {output_image_path}")

            # # Also save as RGB format using PIL (optional, for compatibility)
            # try:
            #     from PIL import Image

            #     vis_rgb = vis_uint8[:, :, ::-1]  # BGR to RGB
            #     rgb_image_path = output_dir / "test_output_rgb.jpg"
            #     Image.fromarray(vis_rgb).save(str(rgb_image_path))
            #     print(f"Saved RGB visualization to: {rgb_image_path}")
            # except ImportError:
            #     pass  # PIL not available, skip RGB save

        # Save raw results
        if len(result["outputs"]) > 0:
            # Use pickle for complex dictionary structures
            results_pkl_path = output_dir / "test_results.pkl"
            save_data = {
                "outputs": result["outputs"],
                "frame_idx": result["frame_idx"],
            }
            with open(str(results_pkl_path), "wb") as f:
                pickle.dump(save_data, f)
            print(f"Saved raw results to: {results_pkl_path}")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_video(video_path: Optional[str] = None):
    """Test with a video file and save rendered mesh video."""
    print("Testing HumanWholeBodyPredictor with video...")

    # Use provided video path or look for example video
    if video_path is None:
        # Look for common video files in example directory
        example_dir = Path(__file__).parent / "example"
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(example_dir.glob(ext)))

        if len(video_files) == 0:
            print(f"No video found in {example_dir}")
            print(
                "Please provide a video path or place a video file in example/ directory"
            )
            return False

        video_path = str(video_files[0])
        print(f"Using video: {video_path}")
    else:
        video_path = str(video_path)

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"Video not found: {video_path}")
        return False

    # Check if checkpoints exist
    checkpoint_path, mhr_path = get_checkpoint_paths()

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please download checkpoints first (see INSTALL.md)")
        return False

    if not mhr_path.exists():
        print(f"MHR model not found: {mhr_path}")
        print("Please download checkpoints first (see INSTALL.md)")
        return False

    try:
        # Initialize predictor
        print("\nInitializing predictor...")
        predictor = HumanWholeBodyPredictor(
            checkpoint_path=str(checkpoint_path),
            mhr_path=str(mhr_path),
            bbox_thresh=0.8,
        )

        # Process video
        print(f"\nProcessing video: {video_path}")
        print("This may take a while depending on video length...")
        results = predictor.predict_from_path(video_path, full_frame=True, verbose=True)

        if len(results) == 0:
            print("ERROR: No results returned!")
            return False

        print(f"\nProcessed {len(results)} frames")

        # Get FPS from first result (should be same for all frames)
        fps = results[0].get("fps", 30.0)
        print(f"Video FPS: {fps:.2f}")

        # Check if we have visualizations
        num_with_vis = sum(1 for r in results if r["visualization"] is not None)
        print(f"Frames with visualization: {num_with_vis}/{len(results)}")

        # Save output video
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)

        if num_with_vis > 0:
            # Get dimensions from first frame with visualization
            first_vis_frame = next(
                (r for r in results if r["visualization"] is not None), None
            )
            if first_vis_frame is not None:
                h, w, _ = first_vis_frame["visualization"].shape

                # Create output video filename
                video_name = video_path_obj.stem
                output_video_path = output_dir / f"test_output_{video_name}.mp4"

                # Use mp4v codec (H.264 compatible)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

                if not out.isOpened():
                    print(f"ERROR: Failed to open video writer for {output_video_path}")
                    return False

                print(f"\nSaving rendered video to: {output_video_path}")
                frames_saved = 0

                for res in results:
                    vis = res["visualization"]
                    if vis is not None:
                        # Ensure uint8 format
                        vis_uint8 = (
                            vis.astype(np.uint8) if vis.dtype != np.uint8 else vis
                        )
                        out.write(vis_uint8)
                        frames_saved += 1

                out.release()
                print(f"Saved {frames_saved} frames to {output_video_path}")
            else:
                print("WARNING: No frames with visualization found!")
        else:
            print("WARNING: No visualizations generated!")

        # Save raw results
        # Note: outputs contain dictionaries with varying shapes, so we use pickle for complex data
        results_pkl_path = output_dir / f"test_results_{video_path_obj.stem}.pkl"
        save_data = {
            "outputs": [res["outputs"] for res in results],
            "frame_indices": [res["frame_idx"] for res in results],
            "fps": fps,
            "video_path": str(video_path),
            "num_frames": len(results),
        }
        with open(str(results_pkl_path), "wb") as f:
            pickle.dump(save_data, f)
        print(f"Saved raw results to: {results_pkl_path}")

        print("\n✓ Video test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Video test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test HumanWholeBodyPredictor with image or video"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "video"],
        default="image",
        help="Test mode: 'image' or 'video' (default: image)",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to video file (required if mode is 'video')",
    )
    args = parser.parse_args()

    if args.mode == "image":
        success = test_single_image()
    elif args.mode == "video":
        success = test_video(args.video_path)
    else:
        print(f"Unknown mode: {args.mode}")
        success = False

    sys.exit(0 if success else 1)
