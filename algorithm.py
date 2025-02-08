import datetime
from ml_predictor import MLModel  # For type hinting if needed

def compute_phase_green_times(roads_counts, total_cycle=120):
    """
    Computes green times for the two phases using real-time car counts.
    For Phase A (north-south): use max(car count in north, car count in south).
    For Phase B (east-west): use max(car count in east, car count in west).

    The green time for each phase is calculated as:
      (phase_max_count / total_car_count) * total_cycle seconds.
    If total_car_count is zero, the cycle is split equally.
    
    Returns a list with two values: [green_time_phase_A, green_time_phase_B].
    """
    north_count = roads_counts.get("north", {}).get("car", 0)
    south_count = roads_counts.get("south", {}).get("car", 0)
    east_count  = roads_counts.get("east", {}).get("car", 0)
    west_count  = roads_counts.get("west", {}).get("car", 0)
    
    total = north_count + south_count + east_count + west_count
    if total == 0:
        return [total_cycle / 2, total_cycle / 2]
    
    phase_A = max(north_count, south_count)
    phase_B = max(east_count, west_count)
    
    green_A = (phase_A / total) * total_cycle
    green_B = (phase_B / total) * total_cycle

    return [green_A, green_B]

def fuzzy_green_time(car_count):
    # Simple fuzzy logic: if count < 10 then 30 sec, if < 20 then 60, else 90.
    if car_count < 10:
        return 30
    elif car_count < 20:
        return 60
    else:
        return 90

def optimize_intersections(traffic_data, prediction_data, config, current_time, rl_agent=None, ml_model=None):
    """
    Enhanced optimization that includes:
      - Dynamic greenlight timing using real-time car counts.
      - Emergency mode: at any intersection, if an ambulance is detected on any road,
        that intersection enters emergency mode. In emergency mode, the entire phase 
        corresponding to the ambulance's location is forced GREEN.
      - Fuzzy logic control (if enabled) in normal mode.
      - ML–optimized (proactive) mode: an ML model predicts the optimal green time based
        on effective vehicle count and current time.
      - School release time adjustment.
      - Adds an overall congestion level for the intersection.
      - Accident detection is reported only (and does not affect the signal).
      - **New:** If one phase has no vehicles while the other has some, force the phase 
        with vehicles.
    Also ensures that one entire phase is always green.
    """
    operation_mode = config.get("operation_mode", "normal")
    use_fuzzy_logic = config.get("use_fuzzy_logic", False)
    results = {}

    # Process each intersection.
    for inter_no, roads in traffic_data.items():
        # Emergency mode: if any road detects an ambulance.
        emergency_detected = False
        emergency_road = None
        for road, counts in roads.items():
            if counts.get("ambulance", 0) > 0:
                emergency_detected = True
                emergency_road = road
                break
        if emergency_detected:
            # Determine emergency phase based on the road where the ambulance was detected.
            emergency_phase = "A" if emergency_road in ["north", "south"] else "B"
            results[inter_no] = {
                "phase": "EMERGENCY",
                "roads": roads,
                "dynamic_duration": 15,
                "emergency_phase": emergency_phase
            }
            continue

        # Compute reactive counts.
        count_A = sum(roads.get(r, {}).get("car", 0) for r in ["north", "south"])
        count_B = sum(roads.get(r, {}).get("car", 0) for r in ["east", "west"])
        pred_A = sum(prediction_data[inter_no].get(r, {}).get("car", 0) for r in ["north", "south"])
        pred_B = sum(prediction_data[inter_no].get(r, {}).get("car", 0) for r in ["east", "west"])
        effective_A = 0.5 * count_A + 0.5 * pred_A
        effective_B = 0.5 * count_B + 0.5 * pred_B

        # --- New forced phase switch logic ---
        # Compute total vehicles for each phase including cars and schoolbuses.
        phase_A_total = sum(roads.get(r, {}).get("car", 0) + roads.get(r, {}).get("schoolbus", 0) 
                            for r in ["north", "south"])
        phase_B_total = sum(roads.get(r, {}).get("car", 0) + roads.get(r, {}).get("schoolbus", 0) 
                            for r in ["east", "west"])
        if phase_A_total == 0 and phase_B_total > 0:
            chosen_phase = "B"
            print(f"Intersection {inter_no}: No vehicles in north-south; switching to Phase B.")
        elif phase_B_total == 0 and phase_A_total > 0:
            chosen_phase = "A"
            print(f"Intersection {inter_no}: No vehicles in east-west; switching to Phase A.")
        else:
            # Decide based on fuzzy logic if enabled in normal mode, else compare effective counts.
            if use_fuzzy_logic and operation_mode == "normal":
                fuzzy_time_A = fuzzy_green_time(effective_A)
                fuzzy_time_B = fuzzy_green_time(effective_B)
                chosen_phase = "A" if fuzzy_time_A >= fuzzy_time_B else "B"
            else:
                chosen_phase = "A" if effective_A >= effective_B else "B"

        # Now compute the reactive_duration.
        base_duration = config.get("base_duration", 10)
        extension_factor = config.get("extension_factor", 0.5)
        max_extension = config.get("max_extension", 20)
        effective_count = effective_A if chosen_phase == "A" else effective_B
        reactive_duration = base_duration + min(effective_count * extension_factor, max_extension)

        # School release time adjustment for intersection "3" on road "west".
        if inter_no == "3":
            if "15:25" <= current_time.strftime("%H:%M") <= "15:35":
                reactive_duration *= 1.5

        if operation_mode == "ml" and ml_model is not None:
            effective_count = effective_A if chosen_phase == "A" else effective_B
            dynamic_duration = ml_model.predict_optimal_green(effective_count, current_time)
        else:
            dynamic_duration = reactive_duration

        results[inter_no] = {
            "phase": chosen_phase,
            "roads": roads,
            "dynamic_duration": dynamic_duration
        }

    # Build final output signals.
    output = []
    for inter_no, data in results.items():
        phase = data["phase"]
        roads = data["roads"]
        dynamic_duration = data["dynamic_duration"]
        # Compute phase green times based on actual car counts.
        lane_green_times = compute_phase_green_times(roads, total_cycle=120)
        total_cars = sum(roads.get(r, {}).get("car", 0) for r in ["north", "south", "east", "west"])
        if total_cars > 50:
            congestion_level = "high"
        elif total_cars > 20:
            congestion_level = "medium"
        else:
            congestion_level = "low"
        for road_no, counts in roads.items():
            if phase == "EMERGENCY":
                emergency_phase = data.get("emergency_phase")
                if emergency_phase == "A":
                    signal = "GREEN" if road_no in ["north", "south"] else "RED"
                else:
                    signal = "GREEN" if road_no in ["east", "west"] else "RED"
            else:
                if phase == "A" and road_no in ["north", "south"]:
                    signal = "GREEN"
                elif phase == "B" and road_no in ["east", "west"]:
                    signal = "GREEN"
                else:
                    signal = "RED"
            # Accident detection is reported but does not change the signal.
            out_item = {
                "intersection": inter_no,
                "road": road_no,
                "cars": counts.get("car", 0),
                "ambulances": counts.get("ambulance", 0),
                "schoolbuses": counts.get("schoolbus", 0),
                "accidents": counts.get("accident", 0),
                "predicted_cars": round(prediction_data[inter_no][road_no]["car"], 1),
                "signal": signal,
                "dynamic_green_duration": round(dynamic_duration, 1),
                "lane_green_times": lane_green_times,
                "congestion_level": congestion_level
            }
            output.append(out_item)
            if counts.get("accident", 0) > 0:
                print(f"ALERT: Accident detected at Intersection {inter_no}, Road {road_no}")

    # If RL mode is chosen, override outputs with RL agent recommendations.
    if operation_mode == "rl" and rl_agent is not None:
        rl_signals = rl_agent.get_optimal_signals(traffic_data, config)
        for inter_id, roads in rl_signals.items():
            for road, rl_data in roads.items():
                for out in output:
                    if out["intersection"] == inter_id and out["road"] == road:
                        out["signal"] = rl_data["signal"]
                        out["dynamic_green_duration"] = rl_data["dynamic_duration"]
                        out["mode"] = "DRL_Optimized"

    # Enforce that one entire phase is active: if phase is "A", then north & south must be GREEN and east & west RED,
    # if phase is "B", then east & west GREEN and north & south RED.
    for inter_no, data in results.items():
        if data.get("phase") == "EMERGENCY":
            continue
        computed_phase = data["phase"]
        for out_item in output:
            if out_item["intersection"] == inter_no:
                if computed_phase == "A":
                    out_item["signal"] = "GREEN" if out_item["road"] in ["north", "south"] else "RED"
                elif computed_phase == "B":
                    out_item["signal"] = "GREEN" if out_item["road"] in ["east", "west"] else "RED"

    return output, {k: v["phase"] for k, v in results.items()}
