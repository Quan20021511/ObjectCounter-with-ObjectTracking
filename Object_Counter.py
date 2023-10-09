import os
import cv2
import numpy as np
from scipy.ndimage import label, find_objects
from motpy import Detection, MultiObjectTracker
from filterpy.kalman import KalmanFilter
from motpy.testing_viz import draw_detection, draw_track


class ObjectCounter:
    def __init__(self, video_path: str, video_resize: float = 0.8, viz_wait_ms: int = 1):
        self.video_path = video_path
        self.video_resize = video_resize
        self.viz_wait_ms = viz_wait_ms

        # Initialize video capture and properties
        self.cap, self.cap_fps = self.read_video_file(self.video_path)
        self.frame_width, self.frame_height = self.get_frame_dimensions()
        self.new_width, self.new_height = self.calculate_resized_dimensions()

        # Initialize tracking and Kalman filter
        self.tracker = self.initialize_tracker()
        self.kalman_filter = self.initialize_kalman_filter()

        # Initialize object counting variables
        self.object_count = 0
        self.tracked_objects = set()

    def read_video_file(self, video_path: str):
        video_path = os.path.expanduser(video_path)
        cap = cv2.VideoCapture(video_path)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        return cap, video_fps

    def get_frame_dimensions(self):
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return frame_width, frame_height

    def calculate_resized_dimensions(self):
        new_width = int(self.frame_width * self.video_resize)
        new_height = int(self.frame_height * self.video_resize)
        return new_width, new_height

    def initialize_tracker(self):
        dt = 1 / self.cap_fps
        model_spec = {'order_pos': 1, 'dim_pos': 2, 'order_size': 0, 'dim_size': 2, 'q_var_pos': 5000.,
                      'r_var_pos': 0.1}
        return MultiObjectTracker(dt=dt, model_spec=model_spec)

    def initialize_kalman_filter(self):
        kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
        return kalman_filter

    def erosion(self, src):
        kernel = np.ones((5, 5), np.uint8)
        erosion_dst = cv2.erode(src, kernel)
        return erosion_dst

    def dilatation(self, src):
        kernel = np.ones((5, 5), np.uint8)
        dilatation_dst = cv2.dilate(src, kernel)
        return dilatation_dst

    def label_array(self, array_binary):
        labeled_array, num_features = label(array_binary)
        return labeled_array, num_features

    def process_image(self, labeled_array, object_id):
        out_detections = []
        slices = find_objects(labeled_array)
        if slices is not None:
            object_counter = 0  # Initialize an object counter
            for label_slice in slices:
                bboxx = label_slice[1]
                bboxy = label_slice[0]
                ymin, ymax, xmin, xmax = bboxy.start, bboxy.stop, bboxx.start, bboxx.stop
                object_counter += 1
                out_detections.append(Detection(box=[xmin, ymin, xmax, ymax], score=1, class_id=object_counter))
        return out_detections

    def run(self):
        object_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.new_width, self.new_height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, threshold1 = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
            threshold_dilation = self.dilatation(threshold1)
            threshold_erosion = self.erosion(threshold_dilation)
            labeled_array, num_features = self.label_array(threshold_erosion)
            detections = self.process_image(labeled_array, object_id)

            for det in detections:
                self.kalman_filter.update(det.box[:2])

            tracks = self.tracker.step(detections)
            active_tracks = self.tracker.active_tracks(min_steps_alive=45)

            for track in active_tracks:
                track_id = track.id
                if track_id not in self.tracked_objects:
                    self.tracked_objects.add(track_id)
                    self.object_count += 1

            cv2.putText(frame, f"Object Count: {self.object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2)

            for track in active_tracks:
                draw_track(frame, track)  # Draw the track's bounding box

            for det in detections:
                draw_detection(frame, det)  # Draw the detection's bounding box

            cv2.imshow('frame', frame)
            c = cv2.waitKey(self.viz_wait_ms)
            if c == ord('q'):
                break
            object_id += 1

        print("Total Object Count:", self.object_count)
        self.cap.release()
        cv2.destroyAllWindows()

    def start(self):
        self.run()


if __name__ == '__main__':
    video_path = r"Your video file path"
    object_counter = ObjectCounter(video_path=video_path)
    object_counter.start()
