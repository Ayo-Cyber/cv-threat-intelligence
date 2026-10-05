"""Validate compact VLM observations and render bounded, non-accusatory prose."""

DESTINATIONS = {
    "pocket": "into a pocket",
    "waistband": "into the waistband",
    "clothing": "beneath clothing",
    "personal_bag": "inside a carried bag",
}
LIMITATIONS = {
    "occluded": "The key action is obscured.",
    "low_light": "Lighting limits visibility.",
    "out_of_frame": "The key action leaves the camera view.",
    "unclear": "The action cannot be established from these frames.",
}


def assess_concealment(data: dict, temporal_frames: int) -> tuple[bool, str]:
    """Indices refer to supplied full frames, never the appended subject crop.

    This validates internal consistency, not the visual truth of model claims.
    Missing observations fail closed without another inference request.
    """
    start, end = data.get("start_frame"), data.get("end_frame")
    sequence = (type(start) is int and type(end) is int
                and 1 <= start < end <= temporal_frames)
    same_subject = data.get("same_subject") is True
    item = data.get("item_visible") is True
    action = data.get("action")
    destination = data.get("destination")
    limitation = data.get("limitation")
    destination_valid = isinstance(destination, str) and destination in DESTINATIONS
    supported = bool(sequence and same_subject and item and action == "insertion"
                     and destination_valid and limitation == "none"
                     and (destination != "pocket" or data.get("opening_visible") is True))
    prefix = "Visual assessment (AI): "
    if supported:
        return True, (prefix + "The person appears to move an item "
                      + DESTINATIONS[destination] + ". Possible product concealment; "
                      "review the recorded sequence.")
    if not sequence:
        return False, prefix + "An ordered item-movement sequence could not be established from the cited frames. Concealment is not established."
    if not same_subject:
        return False, prefix + "The same person and item could not be established across the sequence. Concealment is not established."
    if not item:
        return False, prefix + "A gesture was flagged, but an item could not be established. Concealment is not established."
    descriptions = {
        "removal": "The person appears to retrieve an item rather than conceal it.",
        "holding": "The person appears to hold an item; insertion is not established.",
        "touching": "Contact is visible, but item insertion is not established.",
    }
    description = descriptions.get(action) if isinstance(action, str) else None
    description = description or "Item movement is reported, but its final placement is not established."
    limit = LIMITATIONS.get(limitation, "") if isinstance(limitation, str) else ""
    return False, prefix + description + (" " + limit if limit else "")
