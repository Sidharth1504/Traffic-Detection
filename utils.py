import cv2

def draw_roi(frame, roi, inter_no, road_no, counts, signal, dynamic_duration=None, mode="Normal"):
    x, y, w, h = roi
    # In DRL_Optimized mode, use blue for GREEN and yellow for RED.
    if mode == "DRL_Optimized":
        if signal == "GREEN":
            color = (255, 0, 0)       # Blue
        elif signal == "RED":
            color = (0, 255, 255)     # Yellow
        else:
            color = (255, 0, 255)     # Purple for fallback states
    else:
        # Normal mode: green for GREEN, red for RED.
        if signal == "GREEN":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"I{inter_no} R{road_no}: C{counts['car']} A{counts['ambulance']} S{counts['schoolbus']} Acc{counts['accident']} | {signal} | {mode}"
    if dynamic_duration is not None:
        text += f" DG:{dynamic_duration}s"
    text_y = max(y - 10, 20)
    cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
