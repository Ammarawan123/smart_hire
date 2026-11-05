import cv2
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config
from vision_pipeline_v3 import GazeModel

CAM_INDEX = 0          # agar galat camera khule to 1 try karo
g = GazeModel(str(config.GAZE_ONNX))
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Camera ki taraf seedha dekho, roshni theek rakho... (10 second tak try karega)")
for _ in range(30):                      # warm-up frames
    cap.read()

crop = None
last = None
end = time.time() + 10
while time.time() < end:
    ok, frame = cap.read()
    if not ok:
        continue
    last = frame
    faces = det.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 3, minSize=(80, 80))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        crop = frame[y:y + h, x:x + w]
        break
cap.release()

if crop is None:
    out = config.OUTPUTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    if last is not None:
        cv2.imwrite(str(out / "mirror_debug.jpg"), last)
    print("Chehra nahi mila. outputs\\mirror_debug.jpg dekho: frame kala to nahi? CAM_INDEX badlo.")
else:
    print(g.mirror_selfcheck(crop))