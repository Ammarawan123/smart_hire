"""
check_face_backend.py - one-command check that the V3 'tasks' face backend works on YOUR machine.

  python check_face_backend.py --task face_landmarker.task --image my_face.jpg
  python check_face_backend.py --task face_landmarker.task --webcam        (grabs one frame from camera 0)

Prints: faces found, landmark count (expect 478), the 5 strongest blendshapes, and the AU-like summary.
PASS means the backend ran and returned 478 landmarks plus blendshapes for the face.
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_pipeline_v3 import TasksFaceBackend, summarize_blendshapes  # noqa: E402


def check(backend, rgb) -> dict:
    res = backend.process(rgb)
    faces = res.multi_face_landmarks or []
    out = {"faces": len(faces), "landmarks_first_face": len(faces[0].landmark) if faces else 0, "pass": False}
    if faces:
        bl = res.blendshapes_by_face.get(id(faces[0])) or {}
        out["top5_blendshapes"] = sorted(bl.items(), key=lambda kv: -kv[1])[:5]
        out["au_like_summary"] = summarize_blendshapes(bl)
        out["pass"] = out["landmarks_first_face"] == 478 and bool(bl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image"); g.add_argument("--webcam", action="store_true")
    a = ap.parse_args()
    if a.webcam:
        cap = cv2.VideoCapture(0)
        for _ in range(10):
            ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit("could not read a frame from camera 0")
    else:
        frame = cv2.imread(a.image)
        if frame is None:
            sys.exit(f"cannot read {a.image}")
    be = TasksFaceBackend(a.task)
    for _ in range(3):                      # VIDEO mode: the first frames warm up the tracker
        out = check(be, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    be.close()
    for k, v in out.items():
        print(f"{k}: {v}")
    print("RESULT:", "PASS" if out["pass"] else "FAIL - see numbers above")


if __name__ == "__main__":
    main()
