"""Switch the scenario instructions to the v2 camera-move style (kept for provenance; run once)."""

import json
import os

NEW = {
    "s01_room_forward": "Fast forward dolly shot, first-person view, the camera is moving forward continuously across the living room toward the window on the far wall, passing the dining table on the left, strong forward camera motion, objects get closer and larger every second.",
    "s02_hallway": "Fast forward dolly shot down a hallway, first-person view, the camera is moving forward continuously along the teal rug past the dark dresser on the right, strong forward camera motion, the corridor walls flow past the camera.",
    "s03_doorway": "Forward dolly shot through a doorway, first-person view, the camera moves forward continuously through the open door into the next room, the door frame grows larger and passes around the camera, then the camera slows down inside the room.",
    "s04_turn_left": "Camera pans to the left while moving forward, first-person view, the camera turns continuously to the left, strong left yaw rotation, the dining table sweeps from the center to the right edge of the frame as the left wall comes into view.",
    "s05_ascend": "Crane up shot, first-person view, the camera rises continuously upward toward the ceiling, strong upward camera motion, the bed and furniture move down and out of the bottom of the frame and the ceiling grows larger.",
    "s06_bed_approach": "Forward dolly shot into a bedroom, first-person view, the camera moves forward continuously through the doorway toward the bed, the bed gets closer and larger every second, then the camera stops above the foot of the bed.",
    "s07_turn_right": "Camera pans to the right while moving forward, first-person view, the camera turns continuously to the right, strong right yaw rotation, the round table with black chairs and the plant sweep from the right edge into the center of the frame.",
    "s08_strafe_right": "Trucking shot to the right, first-person view, the camera slides sideways to the right continuously while facing forward, strong lateral camera motion, the room and the shelf move from right to left across the frame.",
    "s09_approach_stop": "Forward dolly shot, first-person view, the camera moves forward toward the wooden desk with the chair, the desk gets closer and larger, then the camera slows down and stops right in front of the desk.",
    "s10_descend": "Forward and downward camera motion, first-person view, the camera moves forward toward the white table and descends continuously, the table top rises in the frame and gets closer, strong downward camera motion.",
}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios.json")
data = json.load(open(path))
data["prompt_style"] = ("v2 camera-move style. v1 (a 'smooth, steady, preserve the room, never pass through' tail) "
                        "produced static videos: s01 optical flow 0.5 px vs 6.8 px with v2.")
data.pop("tail", None)
for s in data["scenarios"]:
    s.setdefault("instruction_v1", s["instruction"])
    s["instruction"] = NEW[s["id"]]
json.dump(data, open(path, "w"), indent=2)
print("updated", len(data["scenarios"]), "scenarios")
