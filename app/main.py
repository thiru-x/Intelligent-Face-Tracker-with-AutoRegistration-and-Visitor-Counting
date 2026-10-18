"""Command-line entry point for the local face visitor application."""

import argparse
import logging

from app.pipeline import VideoPipeline


LOGGER = logging.getLogger("katomaran")


def _camera_index(value):
    try:
        index = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "camera index must be an integer"
        ) from exc
    if index < 0:
        raise argparse.ArgumentTypeError("camera index must be non-negative")
    return index


def build_parser():
    parser = argparse.ArgumentParser(
        description="Detect and recognize visitors from a local video source."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source",
        help="video file path or stream URL; defaults to config.json",
    )
    source_group.add_argument(
        "--camera",
        type=_camera_index,
        help="local camera index, for example --camera 0",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="process video without opening an OpenCV display window",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="serve the browser dashboard on localhost",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="dashboard bind address (loopback addresses only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="dashboard port (default: 8000)",
    )
    return parser


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.web:
        if args.source is not None or args.camera is not None:
            raise SystemExit("--web cannot be combined with --source or --camera")
        try:
            from app.web import run_server

            run_server(host=args.host, port=args.port)
        except (OSError, ValueError):
            LOGGER.exception("Could not start the localhost dashboard")
            return 1
        return 0

    source = args.camera if args.camera is not None else args.source

    try:
        pipeline = VideoPipeline()
        pipeline.process_video(
            video_source=source,
            display=not args.no_display,
        )
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested")
        return 130
    except (FileNotFoundError, RuntimeError, ValueError, OSError):
        LOGGER.exception("Could not run the visitor application")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())