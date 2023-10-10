# Object Counter with Object Tracking

This Python script allows you to count and track objects in a video stream. It can be used for various applications, such as object counting in surveillance videos or monitoring objects on a conveyor belt.

<div align="center">
  <img src="https://github.com/Quan20021511/Digital_Clock/assets/129273695/c9e60288-eb84-4a7f-9769-f04e9abc02a4">
  <img src="https://github.com/Quan20021511/Digital_Clock/assets/129273695/7e529ca9-a387-4a4c-85d3-ca2e7ec1c50a">
</div>

## Features

- Real-time object counting and tracking in video streams.
- Object tracking using Kalman filters for smooth and accurate tracking.
- Object counting based on the number of unique tracked objects.
- Adjustable parameters for object tracking and filtering.
- Visualization of object tracking and bounding boxes in the video.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- SciPy
- `motpy` (Multi-Object Tracking for Python)

## Installation

1. Clone this repository or download the script


2. Install the required dependencies:

```bash

pip install opencv-python numpy scipy motpy

```

## Usage

1. Specify the path to your video file in the `video_path` variable within the script.

2. Run the script:

```bash

python object_counter.py

```


3. Press 'q' to exit the video window when you're done.

## Configuration

You can customize the script by adjusting parameters such as threshold values, object tracking settings, and more. Refer to the script comments for configuration details.

## Contributing

Contributions are welcome! If you have any ideas for improvements or find any issues, please [open an issue](https://github.com/yourusername/object-counter/issues) or submit a pull request.

## Acknowledgments

- This script uses the `motpy` library for multi-object tracking. Visit [motpy on GitHub](https://github.com/wmuron/motpy) for more information.

## Disclaimer

- This script is designed for educational purposes and can be a starting point for more advanced object counting and tracking projects.
