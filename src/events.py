def detect_events(prev_state, current_state, slot_label_mem=None):
    if slot_label_mem is None:
        slot_label_mem = {}

    events = []

    for i in range(len(current_state)):

        if not prev_state[i] and current_state[i]:
            vtype = slot_label_mem.get(i, "UNKNOWN") or "UNKNOWN"
            events.append({
                "action"      : "ENTRY",
                "slot"        : i,
                "vehicle_type": vtype,
            })

        elif prev_state[i] and not current_state[i]:
            vtype = slot_label_mem.get(i, "UNKNOWN") or "UNKNOWN"
            events.append({
                "action"      : "EXIT",
                "slot"        : i,
                "vehicle_type": vtype,
            })

    return events