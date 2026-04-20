"""
Perception Module - Unified Entry Point
=======================================

Supports 3 run modes:
  python main.py capture     → Data collection
  python main.py vio         → VIO localization test
  python main.py detect      → Target detection + 3D position estimation
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="RealSense D435i Perception Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mode descriptions:
  capture   Collect YOLO training data with RealSense camera
  vio       Real-time localization via Visual-Inertial Odometry (custom implementation)
  orbslam   ORB-SLAM3 RGB-D-Inertial localization test
  detect    Target detection + depth-based 3D position estimation
        """,
    )
    parser.add_argument(
        "mode",
        choices=["capture", "vio", "detect", "orbslam"],
        help="Select run mode",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="YOLO model path (used in detect mode)",
    )
    parser.add_argument(
        "--no-imu", action="store_true",
        help="Disable IMU and run in Visual-Only mode (shared by vio / orbslam)",
    )
    parser.add_argument(
        "--pi", action="store_true",
        help="Pi optimization mode: 424x240@15fps, nFeatures=500, viewer OFF (orbslam only)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Headless mode: no GUI, print world-frame (x, y, theta) to terminal (vio only)",
    )

    args, remaining = parser.parse_known_args()

    if args.mode == "capture":
        from data_collection.capture import RealsenseCapture, parse_args
        # Pass remaining args to capture's argparse
        sys.argv = [sys.argv[0]] + remaining
        capture_args = parse_args()
        capture = RealsenseCapture(capture_args)
        capture.start()

    elif args.mode == "vio":
        if args.headless:
            from vio.vio_runner import run_vio_headless
            run_vio_headless(use_imu=(not args.no_imu))
        else:
            from vio.vio_runner import run_vio
            run_vio(use_imu=(not args.no_imu))

    elif args.mode == "orbslam":
        import os
        if args.pi or args.headless:
            os.environ["ORBSLAM_NO_VIEWER"] = "1"
        if args.headless:
            from vio.orbslam_runner import run_orbslam_headless
            run_orbslam_headless(use_imu=(not args.no_imu), pi_mode=args.pi)
        else:
            from vio.orbslam_runner import run_orbslam
            run_orbslam(use_imu=(not args.no_imu), pi_mode=args.pi)

    elif args.mode == "detect":
        print("[DETECT] Target detection + 3D position estimation mode")
        print("[TODO] Detection pipeline not yet implemented")
        print("  → See detection/detector.py, detection/position_estimator.py")


if __name__ == "__main__":
    main()
