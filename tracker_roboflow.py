# tracker_roboflow.py
# Football tracker – Roboflow Workflow edition
# Based on tracker_v2.py, detector swapped to brady-powell/football-player-and-ball-tracker-1785894897417
#
# Red = Offense, Blue = Defense
# Works on Apple Silicon M4 – Roboflow Serverless does the heavy inference

import argparse, json, os, cv2, numpy as np, time, csv
from collections import defaultdict, deque

try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

from metrics import TrackMetrics
from roboflow_detector import run_workflow_frame, parse_workflow_detections

# --- try to import the OC-SORT / BoT-SORT tracker from ultralytics ---
# We still use ultralytics for tracking only (no detection)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    # load a tiny yolo just to get access to the tracker
    _dummy = YOLO("yolov8n.pt")
except Exception:
    ULTRALYTICS_AVAILABLE = False

POSITIONS = {
    '1': ('DL', (180,  0,180)),
    '2': ('LB', (0,165,255)),
    '3': ('DB', (0,220,220)),
    '4': ('OL', (100,100,100)),
    '5': ('WR', (0,255,  0)),
    '6': ('QB', (255,255,  0)),
    '7': ('RB', (255,128,  0)),
    '0': (None, (0,0,0)),
}

# ---- Field mask (same as tracker_v2) ----
def estimate_field_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40]); upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return np.ones(frame.shape[:2], dtype=np.uint8)*255
    c = max(contours, key=cv2.contourArea)
    field_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(field_mask, [c], -1, 255, -1)
    return field_mask

# ---- Simple OC-SORT wrapper via ultralytics ----
# Ultralytics can track with external detections if we feed them in.
# Easiest path for v1: let ultralytics YOLO run in tracking mode with
# our Roboflow boxes injected – or just use BYTETracker-style IoU.
#
# For now: naive IoU tracker – good enough to verify RF boxes, then swap to OC-SORT.

class SimpleIoUTracker:
    def __init__(self, iou_thresh=0.3, max_age=30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}  # id -> {box, age}
    @staticmethod
    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        au = (a[2]-a[0])*(a[3]-a[1]); bu = (b[2]-b[0])*(b[3]-b[1])
        return inter / (au + bu - inter + 1e-6)
    def update(self, dets):
        # dets: list of (x1,y1,x2,y2,conf)
        if not self.tracks:
            out=[]
            for d in dets:
                tid = self.next_id; self.next_id+=1
                self.tracks[tid] = {"box": d[:4], "age":0}
                out.append((*d[:4], tid))
            return out
        # greedy IoU matching
        tids = list(self.tracks.keys())
        tboxes = [self.tracks[i]["box"] for i in tids]
        matches = {}; used_d = set()
        for ti, tb in enumerate(tboxes):
            best_j, best_iou = -1, self.iou_thresh
            for dj, det in enumerate(dets):
                if dj in used_d: continue
                i = self.iou(tb, det[:4])
                if i > best_iou: best_iou, best_j = i, dj
            if best_j >= 0:
                matches[tids[ti]] = best_j; used_d.add(best_j)
        out=[]
        # update matched
        for tid, dj in matches.items():
            box = dets[dj][:4]
            self.tracks[tid] = {"box": box, "age":0}
            out.append((*box, tid))
        # new tracks
        for dj, det in enumerate(dets):
            if dj in used_d: continue
            tid = self.next_id; self.next_id+=1
            self.tracks[tid] = {"box": det[:4], "age":0}
            out.append((*det[:4], tid))
        # age out
        for tid in list(self.tracks.keys()):
            if tid not in matches and tid not in [o[4] for o in out]:
                self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]
        return out

def main():
    ap = argparse.ArgumentParser(description="Football tracker – Roboflow Workflow")
    ap.add_argument("--video", required=True)
    ap.add_argument("--conf", type=float, default=0.25, help="currently set in Workflow, kept for compat")
    ap.add_argument("--px_per_yard", type=float, default=0.0)
    ap.add_argument("--csv", default="tracks.csv")
    ap.add_argument("--use_ocsort", action="store_true", help="use ultralytics OC-SORT if available, else simple IoU")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok: print("Can't open video"); return

    tracker = SimpleIoUTracker()
    metrics = TrackMetrics(fps=fps, px_per_yard=args.px_per_yard)

    # First-frame picker – run RF Workflow
    print("Running Roboflow Workflow on first frame…")
    wf = run_workflow_frame(frame)
    players, balls = parse_workflow_detections(wf)
    print(f"  Workflow returned {len(players)} players, {len(balls)} balls")
    print(f"  Raw keys: {list(wf.keys()) if isinstance(wf, dict) else type(wf)}")
    # If parser found nothing, dump a sample so user can fix keys
    if not players:
        print("\n⚠️  No players parsed – check roboflow_detector.parse_workflow_detections()")
        print("Workflow result sample:", str(wf)[:800])
        return

    # Simple first-frame include picker (click to toggle)
    dets_for_picker = [(x1,y1,x2,y2,conf) for x1,y1,x2,y2,conf,_,_ in players]
    included = [True]*len(dets_for_picker)
    team = [0]*len(dets_for_picker)
    pos = [None]*len(dets_for_picker)

    # … (picker UI same as tracker_v2 – trimmed for brevity in v1, auto-include all)
    # TODO: port full click-to-include picker from tracker_v2

    # init tracker with first-frame boxes
    initial_tracks = tracker.update(dets_for_picker)
    # map tracker IDs back to team/pos – simplified: all included
    track_meta = {tid: {"team":0, "pos":None} for *_, tid in initial_tracks}

    out = cv2.VideoWriter("output_tracked.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (frame.shape[1], frame.shape[0]))

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    csv_rows = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        # ---- Roboflow detect ----
        wf = run_workflow_frame(frame)
        players, balls = parse_workflow_detections(wf)
        dets = [(x1,y1,x2,y2,conf) for x1,y1,x2,y2,conf,_,_ in players]

        # ---- Track ----
        tracks = tracker.update(dets)  # [(x1,y1,x2,y2,tid), …]

        # ---- Metrics update ----
        metrics.update(frame_idx, [
            {"id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            for x1,y1,x2,y2,tid in tracks
        ])

        # ---- Draw ----
        for x1,y1,x2,y2,tid in tracks:
            m = metrics.summary(tid)
            label = f"#{tid}  {m['avg_speed']:.1f} {'mph' if args.px_per_yard else 'px/s'}"
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # ball
        for x1,y1,x2,y2,conf in balls:
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,165,255), 2)
            cv2.putText(frame, "ball", (int(x1), int(y1)-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

        out.write(frame)
        cv2.imshow("tracker_roboflow – Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        # csv log
        for x1,y1,x2,y2,tid in tracks:
            s = metrics.summary(tid)
            csv_rows.append([frame_idx, tid, x1,y1,x2,y2,
                s["avg_speed"], s["max_speed"], s["max_accel"],
                s["distance_ran"], s["displacement"]])

    out.release(); cap.release(); cv2.destroyAllWindows()
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame","track_id","x1","y1","x2","y2",
                    "avg_speed","max_speed","max_accel",
                    "distance_ran","displacement"])
        w.writerows(csv_rows)
    print(f"Saved {args.csv}, video output_tracked.mp4")

if __name__ == "__main__":
    main()
