"""Small, transparent assistance rules for supported object detections."""

from collections import Counter, defaultdict
import re

ROAD_USERS = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "train"}
INDOOR_ITEMS = {"chair", "couch", "dining table", "door", "bed", "toilet", "tv", "laptop", "cell phone"}
COCO_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}


def interpret(detections: list[dict], mode: str) -> dict:
    filtered = detections
    if mode == "navigation":
        filtered = [item for item in detections if item["label"] in ROAD_USERS or item["label"] in {"traffic light", "stop sign", "bench"}]
    elif mode == "indoor":
        filtered = [item for item in detections if item["label"] in INDOOR_ITEMS or item["label"] == "person"]
    by_label: dict[str, Counter] = defaultdict(Counter)
    for item in filtered:
        by_label[item["label"]][item["position"]] += 1
    plurals = {"person": "people", "bus": "buses", "bench": "benches", "knife": "knives"}
    parts = []
    for position in ("left", "center", "right"):
        for label, positions in sorted(by_label.items()):
            count = positions.get(position, 0)
            if not count:
                continue
            noun = label if count == 1 else plurals.get(label, f"{label}s")
            location = {"left": "on the left", "center": "in the center", "right": "on the right"}[position]
            parts.append(f"{count} {noun} {location}")
    summary = "Notice: " + "; ".join(parts) + "." if parts else ("No objects detected above the confidence threshold." if not detections else "No objects matched this assistance mode.")
    lead = max(filtered, key=lambda item: item["confidence"], default=None)
    return {
        "summary": summary,
        "relevant_count": len(filtered),
        "relevant_detections": filtered,
        "lead_detection": lead,
        "rule": {"everyday": "All detected COCO objects", "navigation": "People, common vehicles, traffic lights, stop signs, and benches", "indoor": "People and selected indoor objects"}[mode],
        "notice": "Object labels and image position only. This system does not estimate distance or guarantee safety.",
    }


def route_visual_question(message: str, detections: list[dict], vision_available: bool) -> str | None:
    """Answer supported visual intents deterministically from the current frame only."""
    question = re.sub(r"[^a-z0-9 ]", " ", message.lower())
    question = " ".join(question.split())
    surroundings = any(term in question for term in (
        "surroundings", "around me", "what do you see", "what can you see", "describe the scene",
        "describe what", "explain my", "in front of me", "what is in front", "what s in front",
        "what is there", "what s there", "objects in view", "identify the objects", "recognize the objects",
    ))
    asking_nearest_person = "person" in question or "people" in question or "who" in question
    asking_nearest_person = asking_nearest_person and any(term in question for term in ("nearest", "closest"))
    side_match = re.search(r"\b(left|right)\b", question)
    asking_presence = any(term in question for term in ("is there", "are there", "can you see", "do you see", "where is", "where are"))
    requested_labels = [label for label in sorted(COCO_LABELS) if re.search(rf"\b{re.escape(label)}\b", question)]
    if "people" in question and "person" not in requested_labels:
        requested_labels.append("person")

    if not (surroundings or asking_nearest_person or (asking_presence and (side_match or requested_labels))):
        return None
    if not vision_available:
        return "Please start the camera so I can answer that from a current frame."

    if asking_nearest_person:
        people = [item for item in detections if item["label"].lower() == "person"]
        if not people:
            return "I don't see a person in the latest camera frame."
        if len(people) == 1:
            position = people[0]["position"]
            return f"One person is visible on the image {position}. I can't determine physical distance from this camera."
        return f"I can see {len(people)} people, but I can't reliably tell which one is physically closest using this camera."

    if asking_presence and (side_match or requested_labels):
        side = side_match.group(1) if side_match else None
        on_side = [item for item in detections if side is None or item["position"] == side]
        if requested_labels:
            matches = [item for item in on_side if item["label"].lower() in requested_labels]
            label = requested_labels[0]
            article = "an" if label[:1] in "aeiou" else "a"
            if matches:
                if side:
                    return f"Yes, I can see {article} {label} on the image's {side}."
                locations = ", ".join(sorted({item["position"] for item in matches}, key=("left", "center", "right").index))
                return f"Yes, I can see {article} {label} in the image: {locations}."
            anywhere = any(item["label"].lower() == label for item in detections)
            if anywhere:
                location_text = f"on the image's {side}" if side else "in the requested location"
                return f"I can see {article} {label}, but not {location_text}."
            location = f" on the image's {side}" if side else " in the latest frame"
            return f"I don't see {article} {label}{location}."
        if not side:
            return None
        if on_side:
            labels = ", ".join(item["label"] for item in sorted(on_side, key=lambda item: item["box"][0]))
            return f"Yes. On the image's {side}, I can see {labels}."
        return f"I don't see a recognized object on the image's {side} in the latest frame."

    if not detections:
        return "I don't see any recognized objects in the latest camera frame."
    ordered = sorted(detections, key=lambda item: (item["box"][0] + item["box"][2]) / 2)
    position_words = {"left": "on your left", "center": "in the center", "right": "on your right"}
    descriptions = [f"a {item['label']} {position_words.get(item['position'], 'in the image')}" for item in ordered]
    if len(descriptions) == 1:
        scene = descriptions[0]
    elif len(descriptions) == 2:
        scene = f"{descriptions[0]} and {descriptions[1]}"
    else:
        scene = ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
    return f"I can see {scene}. Left and right refer to the image."
