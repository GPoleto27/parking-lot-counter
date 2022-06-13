import cv2
from numpy import array, delete, arange
from console_logging.console import Console

import argparse
from os.path import exists

from utils import *


def main() -> None:
    parser = argparse.ArgumentParser(description="Parking lot vehicle counter")
    parser.add_argument("--video", type=str, default="videos/parking-lot.mp4")
    parser.add_argument("--video_output", type=str, default="videos/output.mp4")
    parser.add_argument("--graph_output", type=str, default="output.png")
    parser.add_argument("--threshold", type=int, default=20)
    parser.add_argument("--start_frame", type=int, default=667)
    parser.add_argument("--rois", type=str, default="rois.txt")
    args = parser.parse_args()

    console = Console()

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        console.error("Could not open video file")
        return

    width, height, fps = get_video_properties(cap)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    ret, frame = cap.read()

    # Save processed frame to showcase the results
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.png", gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imwrite("blur.png", blur)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5
    )
    cv2.imwrite("thresh.png", thresh)

    if not exists(args.rois):
        console.error("Could not find ROIs file")
        console.info("Creating empty ROIs file")
        if not create_rois(args.rois):
            console.error("Could not create ROIs file")
            return
        console.success("Created empty ROIs file")

    loaded_rois = load_rois(args.rois)
    console.success("Loaded ROIs")
    draw_boxes(frame, loaded_rois)

    rois = list(cv2.selectROIs("Select ROIs", frame, True))
    rois.extend(loaded_rois)
    save_rois(rois, args.rois)
    console.success("Saved ROIs")

    rois = array(rois)
    total = len(rois)

    cv2.destroyAllWindows()

    frames = []
    count_series = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            console.error("Could not read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5
        )

        slices = array(
            [thresh[y : y + h, x : x + w] for x, y, w, h in rois], dtype=object
        )

        occupied_idxs = get_occupied_rois(slices, args.threshold)
        free_idxs = delete(arange(total), occupied_idxs)

        count = len(free_idxs)

        draw_boxes(frame, rois[occupied_idxs], (0, 0, 255))
        draw_boxes(frame, rois[free_idxs], (0, 255, 0))
        cv2.putText(
            frame,
            f"Vagas: {count}/{total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Parking Lot Counter", frame)

        frames.append(frame)
        count_series.append(count)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

    console.success("Processed video")
    cap.release()
    cv2.destroyAllWindows()

    console.info(f"Saving video to {args.video_output}:{width}x{height}@{fps}")
    write_video(frames, args.video_output, fps, width, height)
    console.success(f"Saved video to {args.video_output}:{width}x{height}@{fps}")

    console.info(f"Saving graph to {args.graph_output}")
    plot_time_series(count_series, "Série Temporal", "Frame", "Vagas", args.graph_output)
    console.success(f"Saved graph to {args.graph_output}")


if __name__ == "__main__":
    main()
