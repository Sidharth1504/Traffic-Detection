import datetime

def optimize_intersections(traffic_data, prediction_data, config, current_time, rl_agent=None):
    """
    Advanced optimization that fuses rule‑based dynamic signal timing with optional DRL‑based optimization.
    If an rl_agent is provided (and DRL mode is active), its recommendations override normal decisions.
    """
    results = {}
    school_bus_time = config.get("school_bus_time", "15:00")
    school_intersection = config.get("school_intersection", "1")

    # First, process each intersection using rule‑based logic.
    for inter_no, roads in traffic_data.items():
        accident_flag = any(roads.get(road, {}).get("accident", 0) > 0 for road in roads)
        if accident_flag:
            base_duration = config.get("base_duration", 10)
            results[inter_no] = {"phase": "ACCIDENT", "roads": roads, "dynamic_duration": base_duration}
            continue

        count_A = sum(roads.get(road, {}).get("car", 0) for road in ["north", "south"])
        count_B = sum(roads.get(road, {}).get("car", 0) for road in ["east", "west"])
        pred_A = sum(prediction_data[inter_no].get(road, {}).get("car", 0) for road in ["north", "south"])
        pred_B = sum(prediction_data[inter_no].get(road, {}).get("car", 0) for road in ["east", "west"])
        effective_A = 0.5 * count_A + 0.5 * pred_A
        effective_B = 0.5 * count_B + 0.5 * pred_B

        ambulance_A = sum(roads.get(road, {}).get("ambulance", 0) for road in ["north", "south"])
        ambulance_B = sum(roads.get(road, {}).get("ambulance", 0) for road in ["east", "west"])
        current_time_str = current_time.strftime("%H:%M")
        is_school_bus_time = (current_time_str == school_bus_time) and (inter_no == school_intersection)

        if ambulance_A > 0:
            chosen_phase = "A"
        elif ambulance_B > 0:
            chosen_phase = "B"
        elif is_school_bus_time:
            schoolbus_A = sum(roads.get(road, {}).get("schoolbus", 0) for road in ["north", "south"])
            schoolbus_B = sum(roads.get(road, {}).get("schoolbus", 0) for road in ["east", "west"])
            chosen_phase = "A" if schoolbus_A >= schoolbus_B else "B"
        else:
            chosen_phase = "A" if effective_A >= effective_B else "B"

        base_duration = config.get("base_duration", 10)
        extension_factor = config.get("extension_factor", 0.5)
        effective_count = effective_A if chosen_phase == "A" else effective_B
        max_extension = config.get("max_extension", 20)
        dynamic_duration = base_duration + min(effective_count * extension_factor, max_extension)

        results[inter_no] = {"phase": chosen_phase, "roads": roads, "dynamic_duration": dynamic_duration}

    # Build final output signals.
    output = []
    for inter_no, data in results.items():
        phase = data["phase"]
        roads = data["roads"]
        dynamic_duration = data["dynamic_duration"]
        for road_no, counts in roads.items():
            if phase == "ACCIDENT":
                signal = "FLASHING RED"
            elif phase == "A" and road_no in ["north", "south"]:
                signal = "GREEN"
            elif phase == "B" and road_no in ["east", "west"]:
                signal = "GREEN"
            else:
                signal = "RED"
            output.append({
                "intersection": inter_no,
                "road": road_no,
                "cars": counts.get("car", 0),
                "ambulances": counts.get("ambulance", 0),
                "schoolbuses": counts.get("schoolbus", 0),
                "accidents": counts.get("accident", 0),
                "predicted_cars": round(prediction_data[inter_no][road_no]["car"], 1),
                "signal": signal,
                "dynamic_green_duration": round(dynamic_duration, 1)
                # "mode" will be set later.
            })
            if counts.get("accident", 0) > 0:
                print(f"ALERT: Accident detected at Intersection {inter_no}, Road {road_no}")

    # Override with DRL recommendations if an rl_agent is provided.
    if rl_agent is not None:
        rl_signals = rl_agent.get_optimal_signals(traffic_data, config)
        for inter_id, roads in rl_signals.items():
            for road, rl_data in roads.items():
                for out in output:
                    if out["intersection"] == inter_id and out["road"] == road:
                        out["signal"] = rl_data["signal"]
                        out["dynamic_green_duration"] = rl_data["dynamic_duration"]
                        out["mode"] = "DRL_Optimized"

    # *** NEW CODE: Force at least one road green per intersection ***
    intersections_set = set([item["intersection"] for item in output])
    for inter_no in intersections_set:
        # Skip if intersection is in ACCIDENT mode.
        if results.get(inter_no, {}).get("phase") == "ACCIDENT":
            continue
        # Gather signals for this intersection.
        signals = [item["signal"] for item in output if item["intersection"] == inter_no]
        # If no road is green, force one road to be green.
        if "GREEN" not in signals:
            # Try to force "north" to be green if available.
            forced = False
            for item in output:
                if item["intersection"] == inter_no and item["road"] == "north":
                    item["signal"] = "GREEN"
                    forced = True
                    break
            # If "north" is not available, force the first road.
            if not forced:
                for item in output:
                    if item["intersection"] == inter_no:
                        item["signal"] = "GREEN"
                        break

    return output, {k: v["phase"] for k, v in results.items()}
