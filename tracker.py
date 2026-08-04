#!/usr/bin/env python3
"""
Football 22-player tracker
Red = Offense, Blue = Defense
Click to include/exclude before play starts.
"""
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import json
import os

# ---------- Field mask ----------
def estimate_field_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # green field range – tune if needed
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones(frame.shape[:2], dtype=np.uint8) * 255
    c = max(contours, key=cv2.contourArea)
    field_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(field_mask, [c], -1, 255, -1)
    return field_mask

def draw_field_polygon_interactive(frame):
    pts = []
    clone = frame.copy()
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(clone, (x,y), 4, (0,255,255), -1)
            if len(pts) > 1:
                cv2.line(clone, pts[-2], pts[-1], (0,255,255), 2)
            cv2.imshow("Draw field - click corners, C to close", clone)
    cv2.imshow("Draw field - click corners, C to close", clone)
    cv2.setMouseCallback("Draw field - click corners, C to close", on_click)
    while True:
        cv2.imshow("Draw field - click corners, C to close", clone)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('c') and len(pts) >= 3:
            break
    cv2.destroyWindow("Draw field - click corners, C to close")
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    return mask, pts

# ---------- Selection UI ----------
class Selector:
    def __init__(self, frame, detections):
        # detections: list of [x1,y1,x2,y2,conf]
        self.frame = frame.copy()
        self.dets = detections
        self.included = [True] * len(detections)
        self.team = [0] * len(detections)  # 0=offense/red, 1=defense/blue
        self.win = "Select Players - Click toggle | R=Offense B=Defense | SPACE=Go Q=Quit"

    def point_in_box(self, x, y, box):
        x1,y1,x2,y2 = box[:4]
        return x1 <= x <= x2 and y1 <= y <= y2

    def draw(self):
        vis = self.frame.copy()
        for i, d in enumerate(self.dets):
            x1,y1,x2,y2 = map(int, d[:4])
            inc = self.included[i]
            is_off = self.team[i] == 0
            color = (0,0,255) if is_off else (255,120,0)
            if not inc:
                color = (120,120,120)
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            label = f"{'OFF' if is_off else 'DEF'} {i}{' -OUT' if not inc else ''}"
            cv2.putText(vis, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        info = f"Included: {sum(self.included)}/{len(self.dets)}  [R/B click to set team]"
        cv2.putText(vis, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(vis, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        return vis

    def run(self):
        current_team_key = [0]  # mutable
        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for i, d in enumerate(self.dets):
                    if self.point_in_box(x,y,d):
                        # shift-click toggles team, normal click toggles include
                        if flags & cv2.EVENT_FLAG_SHIFTKEY:
                            self.team[i] ^= 1
                        else:
                            self.included[i] = not self.included[i]
                        break
        cv2.namedWindow(self.win)
        cv2.setMouseCallback(self.win, mouse)
        while True:
            cv2.imshow(self.win, self.draw())
            k = cv2.waitKey(30) & 0xFF
            if k == ord('r'):  # mass set selected? just hint – individual is shift-click
                pass
            if k == ord('q'):
                cv2.destroyWindow(self.win)
                return None
            if k == ord(' '):
                break
            if k == ord('r') & 0xFF == 0: pass
            # R/B keys set team for hovered – simpler: r/b apply to all included
            if k == ord('r'):
                for i, inc in enumerate(self.included):
                    if inc: self.team[i] = 0
            if k == ord('b'):
                for i, inc in enumerate(self.included):
                    if inc: self.team[i] = 1
        cv2.destroyWindow(self.win)
        kept_boxes = [self.dets[i] for i, inc in enumerate(self.included) if inc]
        kept_teams = [self.team[i] for i, inc in enumerate(self.included) if inc]
        return kept_boxes, kept_teams

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--tracker", default="ocsort.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--draw_field", action="store_true", help="manually draw field polygon")
    ap.add_argument("--field_json", default=None, help="load/save field polygon")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, first_frame = cap.read()
    if not ok:
        print("Can't open video"); return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Field mask
    if args.field_json and os.path.exists(args.field_json):
        poly = json.load(open(args.field_json))
        mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
        print(f"Loaded field polygon from {args.field_json}")
    elif args.draw_field:
        mask, poly = draw_field_polygon_interactive(first_frame)
        if args.field_json:
            json.dump(poly, open(args.field_json, "w"))
    else:
        mask = estimate_field_mask(first_frame)
        print("Field mask auto-estimated (use --draw_field if sidelines leak in)")

    model = YOLO(args.model)

    # First-frame detect for selection
    res = model.predict(first_frame, classes=[0], conf=args.conf, iou=args.iou_thresh, imgsz=args.imgsz, verbose=False)[0]
    dets = []
    h, w = first_frame.shape[:2]
    for b in res.boxes:
        x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
        cx = int((x1+x2)/2); cy = int((y1+y2)/2)
        if cy < 0 or cy >= h or cx < 0 or cx >= w: continue
        if mask[cy, cx] == 0:  # outside field
            continue
        conf = float(b.conf[0])
        dets.append([x1,y1,x2,y2,conf])
    # keep largest 30 in-field, sorted left->right to make selection sane
    dets = sorted(dets, key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)[:30]
    dets = sorted(dets, key=lambda d: d[0])

    if not dets:
        print("No players found in field mask – try --draw_field"); return

    selector = Selector(first_frame, dets)
    sel = selector.run()
    if sel is None:
        print("Quit"); return
    kept_boxes, kept_teams = sel
    print(f"Tracking {len(kept_boxes)} players")

    # Build initial track hints: map detection boxes to team color
    # We'll color by track ID using a nearest-box vote from the selection
    init_map = [(b, t) for b, t in zip(kept_boxes, kept_teams)]

    def box_center(b): return np.array([ (b[0]+b[2])/2, (b[1]+b[3])/2 ])
    init_centers = [box_center(b) for b, _ in init_map]
    init_teams = [t for _, t in init_map]

    assigned_track_teams = {}  # track_id -> 0/1

    # video out
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out_w, out_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_tracked.mp4", fourcc, fps, (out_w, out_h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        res = model.track(frame, classes=[0], conf=args.conf, iou=args.iou_thresh,
                          persist=True, tracker=args.tracker, imgsz=args.imgsz, verbose=False)[0]
        
        if res.boxes is not None and res.boxes.id is not None:
            xyxys = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            for xyxy, tid in zip(xyxys, ids):
                x1,y1,x2,y2 = map(int, xyxy)
                cx = (x1+x2)//2; cy = (y1+y2)//2
                if cy < 0 or cy >= mask.shape[0] or cx < 0 or cx >= mask.shape[1] or mask[cy,cx]==0:
                    continue  # sideline filter every frame
                # assign team on first sight by nearest init selection box
                if tid not in assigned_track_teams:
                    ctr = np.array([cx, cy])
                    if init_centers:
                        dists = [np.linalg.norm(ctr - ic) for ic in init_centers]
                        assigned_track_teams[tid] = init_teams[int(np.argmin(dists))]
                    else:
                        assigned_track_teams[tid] = 0
                team = assigned_track_teams[tid]
                color = (0,0,255) if team == 0 else (255,100,0)  # BGR: red / blue
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"{'OFF' if team==0 else 'DEF'} #{tid}", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        out.write(frame)
        cv2.imshow("Tracking - Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cap.release(); out.release(); cv2.destroyAllWindows()
    print("Saved output_tracked.mp4")

if __name__ == "__main__":
    main()
