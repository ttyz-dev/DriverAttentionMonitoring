import time
import cv2
import obd
import torch
import pygame
import os
from datetime import datetime
from ultralytics import YOLO

# YOLOv8 initialization
model = YOLO("./yolov8/runs/detect/train2/weights/best.pt")
class_names = {
    0: "DangerousDriving",
    1: "Distracted",
    2: "Drinking",
    3: "SafeDriving",
    4: "SleepyDriving",
    5: "Yawn",
}

# Dictionary to keep track detection times for each class
detection_timers = {}
start_times = {}
last_seen_times = {}
durations = {}


# Function to generate dynamic filename
def generate_filename():
    current_date = datetime.now().strftime("%Y%m%d")
    files_today = len(
        [
            name
            for name in os.listdir("logs")
            if os.path.isfile(os.path.join("logs", name))
            and name.startswith(current_date)
        ]
    )
    session_number = files_today + 1
    return os.path.join("logs", f"{current_date}_{session_number}.txt")


start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_file = generate_filename()

# Pygame mixer initialization
pygame.init()
DANGER_SOUND_PATH = "./resource/audio/DANGER.wav"
danger_sound = pygame.mixer.Sound(DANGER_SOUND_PATH)
FOCUS_SOUND_PATH = "./resource/audio/focus.wav"
focus_sound = pygame.mixer.Sound(FOCUS_SOUND_PATH)


# Function to check for vehicle movement using OBD-II data
def is_vehicle_moving(obd_connection):
    # fetch OBD-II speed command
    response = obd_connection.query(obd.commands.SPEED)
    if not response.is_null():
        speed = response.value.magnitude
        if speed > 0:
            return True, speed
    return False, 0


# Function to check if vehicle is powering off
def is_vehicle_powering_off(obd_connection):
    # fetch OBD-II RPM command
    response = obd_connection.query(obd.commands.RPM)
    if response.is_null() or response.value.magnitude == 0:
        return True
    return False


# Function to process frames from model results
def process_frame(results):
    global detection_timers  # access global dictionary
    current_time = time.time()

    with open(log_file, "w") as f:
        for result in results:
            if result.boxes.cls.numel() > 0:
                result_class = torch.tensor(result.boxes.cls)
                for cls in result_class:
                    class_label = class_names[int(cls.item())]
                    match class_label:
                        case "DangerousDriving" | "SleepyDriving":
                            if "danger" not in detection_timers:
                                detection_timers["danger"] = current_time
                            elif current_time - detection_timers["danger"] >= 2:
                                if not pygame.mixer.get_busy():
                                    print(detection_timers)
                                    danger_sound.play()

                        case "Distracted" | "Drinking":
                            if "unfocus" not in detection_timers:
                                detection_timers["unfocus"] = current_time
                            elif current_time - detection_timers["unfocus"] >= 2:
                                if not pygame.mixer.get_busy():
                                    print(detection_timers)
                                    focus_sound.play()
                        case "Yawn":
                            pass
                        case _:
                            # If class label is "SafeDriving" for 2 sec consecutively, reset the detection_timers
                            if "danger" or "drinking" in detection_timers:
                                if "safe" not in detection_timers:
                                    detection_timers["safe"] = current_time
                                elif current_time - detection_timers["safe"] >= 2:
                                    detection_timers.clear()

                    if (
                        class_label not in last_seen_times
                        or current_time - last_seen_times[class_label] > 1
                    ):
                        start_times[class_label] = current_time
                        durations[class_label] = 0

                    last_seen_times[class_label] = current_time

                for class_name in last_seen_times:
                    if current_time - last_seen_times[class_name] > 1:
                        # Calculate the duration and write into the log file
                        duration = current_time - start_times[class_name]
                        durations[class_name] += duration
                        f.write(
                            f"Session Start Time: {start_time}\n\n{class_name}: {durations[class_name]:.2f} seconds"
                        )

                        # reset the last seen time to avoid multiple logs for brief disappearance
                        last_seen_times[class_name] = float("inf")

    return results[0].plot()


# Main Python application logic
def main():
    # setup OBD-II connection
    obd_connection = obd.OBD()

    # video_path = "./resource/video_dataset/gA_1_s1_ir_face.mp4"
    # Video capture setup
    # cap = cv2.VideoCapture(video_path)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Read frame from the video
        success, frame = cap.read()

        # fetch vehicle moving status and speed
        vehicle_status, vehicle_speed = is_vehicle_moving(obd_connection)

        if vehicle_status:
            if success:
                # Process the frame and set a confidence threshold of 50%
                results = model(frame, conf=0.5)

                # Display vehicle speed on frame
                if vehicle_speed is not None:
                    speed_text = f"Speed: {vehicle_speed} km/h"
                    # Draw black background rectangle
                    cv2.rectangle(frame, (0, 0), (100, 50), (0, 0, 0), -1)
                    # Add speed text
                    cv2.putText(
                        frame,
                        speed_text,
                        (5, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (255, 255, 255),
                        2,
                    )
                annotated_frame = process_frame(results)

                # Display the annotated frame
                cv2.imshow("Driver Attention Monitoring", annotated_frame)

                # Break the loop if 'q' is pressed
                if (cv2.waitKey(1) & 0xFF == ord("q")) | is_vehicle_powering_off(
                    obd_connection
                ):
                    """if cv2.waitKey(1) & 0xFF == ord("q"):
                    Log any remaining durations
                    The log file will start with a Session Start Time header
                    And end with a Session End Time, follow by the Total Driving Duration
                    """
                    with open(log_file, "a") as f:
                        f.write(f"Session Start Time: {start_time}\n\n")
                        for class_name, duration in durations.items():
                            if duration >= 1:
                                f.write(f"{class_name}: {duration:.2f} seconds\n")
                        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        total_duration = datetime.strptime(
                            end_time, "%Y-%m-%d %H:%M:%S"
                        ) - datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                        f.write(f"\n\nSession End Time: {end_time}\n")
                        f.write(f"Total Driving Duration: {total_duration}")
                    break
            else:
                # Break the loop if the end of the video is reached
                break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
