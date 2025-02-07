import cv2
import json
import time
import argparse
import datetime
from model import VehicleDetector
from algorithm import optimize_intersections
from utils import draw_roi, draw_detections

def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

def compute_intersections_from_grid(grid_config, frame_width, frame_height):
    intersections = {}
    rows = grid_config["rows"]
    cols = grid_config["cols"]
    roi_width = grid_config["roi_width"]
    roi_height = grid_config["roi_height"]
    start_x = grid_config["start_x"]
    start_y = grid_config["start_y"]

    cell_width = frame_width / cols
    cell_height = frame_height / rows

    inter_id = 1
    for row in range(rows):
        for col in range(cols):
            base_x = col * cell_width
            base_y = row * cell_height
            intersections[str(inter_id)] = {
                "roads": {
                    "north": [int(base_x + cell_width / 2 - roi_width / 2),
                              int(base_y + cell_height / 4 - roi_height / 2), roi_width, roi_height],
                    "south": [int(base_x + cell_width / 2 - roi_width / 2),
                              int(base_y + 3 * cell_height / 4 - roi_height / 2), roi_width, roi_height],
                    "east": [int(base_x + 3 * cell_width / 4 - roi_width / 2),
                             int(base_y + cell_height / 2 - roi_height / 2), roi_width, roi_height],
                    "west": [int(base_x + cell_width / 4 - roi_width / 2),
                             int(base_y + cell_height / 2 - roi_height / 2), roi_width, roi_height]
                }
            }
            inter_id += 1
    return intersections

def main():
    parser = argparse.ArgumentParser(description="Smart Traffic Management System")
    parser.add_argument("--output", choices=["video", "json"], default="video",
                        help="Output method: 'video' for displaying video output, 'json' for json output only")
    args = parser.parse_args()

    config = load_config("config.json")
    video_path = "data/sample_video8.mp4"
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(1)  # Changed from video path to default camera
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if "grid" in config:
        intersections_config = compute_intersections_from_grid(config["grid"], frame_width, frame_height)
    else:
        intersections_config = config.get("intersections", {})

    detector = VehicleDetector()

    scale_factor = 2
    min_phase_duration = config.get("min_phase_duration", 5)
    last_phase_state = {}
    last_phase_switch_time = {}
    config["last_school_bus_green"] = config.get("last_school_bus_green", datetime.datetime.now())

    output_data = []  # To store data for JSON output
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        traffic_data = {}
        for inter_no, inter_data in intersections_config.items():
            roads_config = inter_data.get("roads", {})
            traffic_data[inter_no] = {}
            for road_no, roi in roads_config.items():
                x, y, w, h = [int(coord * scale_factor) for coord in roi]

                if w <= 0 or h <= 0 or y < 0 or x < 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
                    print(f"Skipping invalid ROI for Intersection {inter_no}, Road {road_no}")
                    traffic_data[inter_no][road_no] = {"car": 0, "ambulance": 0, "schoolbus": 0, "accident": 0}
                    continue

                roi_frame = frame[y:y + h, x:x + w]

                if roi_frame.size == 0:
                    print(f"Empty ROI for Intersection {inter_no}, Road {road_no}")
                    traffic_data[inter_no][road_no] = {"car": 0, "ambulance": 0, "schoolbus": 0, "accident": 0}
                    continue

                detections = detector.detect_vehicles(roi_frame)
                counts = {"car": 0, "ambulance": 0, "schoolbus": 0, "accident": 0}
                for detection in detections:
                    counts[detection['class']] += 1
                traffic_data[inter_no][road_no] = counts
                draw_detections(roi_frame, detections)
                frame[y:y + h, x:x + w] = roi_frame

        current_time = datetime.datetime.now()
        output_signals, computed_phases = optimize_intersections(traffic_data, config, current_time)

        current_time = time.time()
        final_phases = {}
        for inter_no, new_phase in computed_phases.items():
            if inter_no not in last_phase_state:
                last_phase_state[inter_no] = new_phase
                last_phase_switch_time[inter_no] = current_time
                final_phases[inter_no] = new_phase
            else:
                prev_phase = last_phase_state[inter_no]
                if new_phase != prev_phase:
                    if current_time - last_phase_switch_time[inter_no] >= min_phase_duration:
                        last_phase_state[inter_no] = new_phase
                        last_phase_switch_time[inter_no] = current_time
                        final_phases[inter_no] = new_phase
                    else:
                        final_phases[inter_no] = prev_phase
                else:
                    final_phases[inter_no] = prev_phase

        final_output_signals = []
        for inter_no, inter_data in intersections_config.items():
            phase = final_phases.get(inter_no, "A")
            roads_config = inter_data.get("roads", {})
            for road_no, roi in roads_config.items():
                counts = traffic_data[inter_no].get(road_no, {"car": 0, "ambulance": 0, "schoolbus": 0, "accident": 0})
                if phase == "A" and road_no in ["north", "south"]:
                    signal = "GREEN"
                elif phase == "B" and road_no in ["east", "west"]:
                    signal = "GREEN"
                else:
                    signal = "RED"
                final_output_signals.append({
                    "intersection": inter_no,
                    "road": road_no,
                    "cars": counts["car"],
                    "ambulances": counts["ambulance"],
                    "schoolbuses": counts["schoolbus"],
                    "signal": signal
                })
                if counts["accident"] > 0:
                    print(f"ALERT: Accident detected at Intersection {inter_no}, Road {road_no}")

        if args.output == "json":
            output_data.append(final_output_signals)
        elif args.output == "video":
            for inter_no, inter_data in intersections_config.items():
                roads_config = inter_data.get("roads", {})
                for road_no, roi in roads_config.items():
                    x, y, w, h = [int(coord * scale_factor) for coord in roi]
                    counts = traffic_data[inter_no].get(road_no, {"car": 0, "ambulance": 0, "schoolbus": 0, "accident": 0})
                    decision = next((item for item in final_output_signals
                                     if item["intersection"] == inter_no and item["road"] == road_no), None)
                    signal = decision["signal"] if decision else "UNKNOWN"
                    draw_roi(frame, (x, y, w, h), inter_no, road_no, counts, signal)

            cv2.imshow("Smart Traffic Management System", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cap.release()
    if args.output == "json":
        with open("traffic_data.json", "w") as f:
            json.dump(output_data, f, indent=4)
    elif args.output == "video":
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()