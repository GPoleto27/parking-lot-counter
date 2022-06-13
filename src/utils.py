import cv2
from numpy import array, where, mean
from matplotlib import pyplot as plt


def get_video_properties(cap: cv2.VideoCapture) -> tuple:
    """
    Gets the video properties
    Args:
        cap: Video capture
    Returns:
        width: Video width
        height: Video height
        fps: Video fps
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    return width, height, fps


def write_video(frames: list, output: str, fps: float, width: int, height: int) -> None:
    """
    Writes a video from a list of frames
    Args:
        frames: List of frames
        output: Output video path
        fps: Video fps
        width: Video width
        height: Video height
    """
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def create_rois(path: str) -> bool:
    """
    Creates a ROIs file
    Args:
        path: ROIs file path
    """
    with open(path, "w") as f:
        return f is not None


def load_rois(path: str) -> list:
    """
    Loads ROIs from a file
    Args:
        path: Path to the file
    Returns:
        rois: List of ROIs
    """
    loaded_rois = []
    with open(path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            x, y, w, h = line.split(",")
            loaded_rois.append((int(x), int(y), int(w), int(h)))
    return loaded_rois


def save_rois(rois: list, path: str) -> None:
    """
    Saves ROIs to a file
    Args:
        rois: List of ROIs
        path: Path to the file
    """
    with open(path, "w") as f:
        for x, y, w, h in rois:
            f.write(f"{x},{y},{w},{h}\n")


def draw_boxes(frame, boxes, color=(255, 0, 0)):
    """
    Draws boxes on a frame
    Args:
        frame: Frame to draw on
        boxes: List of boxes
        color: Color of the boxes
    """
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def get_occupied_rois(slices: array, threshold: float) -> array:
    """
    Gets the occupied ROIs (Parking Spots) from a list of slices
    Args:
        slices: List of slices (Parking Spots)
        threshold: Threshold to consider a slice occupied
    Returns:
        occupied_rois: Indices of occupied ROIs
    """
    occupied = array([mean(s) > threshold for s in slices])
    return where(occupied == True)


def plot_time_series(
    time_series: list, title: str, xlabel: str, ylabel: str, output: str
) -> None:
    """
    Plots a time series
    Args:
        time_series: List of time series
        title: Plot title
        xlabel: X axis label
        ylabel: Y axis label
        output: Output path
    """
    plt.plot(time_series)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output)
    plt.close()
    plt.plot(time_series)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output)
    plt.show()
