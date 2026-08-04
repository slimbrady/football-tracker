#!/usr/bin/env python3
"""
Football 22-player tracker v2
Red = Offense, Blue = Defense
- Jersey number OCR (easyocr, optional)
- Speed + trails
- Ball tracker (COCO sports ball class 32)
- Measure distance between 2 players (D key, click 2)
- Position tagging + position filters: DL, LB, DB, OL, WR, QB, RB
  Keys during selection: 1=DL 2=LB 3=DB 4=OL 5=WR 6=QB 7=RB 0=clear
  During tracking: F1..F7 filter to that position group, F0 show all
  Combo filters: G = DL+LB, H = Secondary only, J = WR vs CB matchup mode
"""
import argparse, json, os, cv2, numpy as np, time, csv
from collections import defaultdict, deque
from ultralytics import YOLO

# --- optional OCR ---
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

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
POS_NAME_TO_KEY = {v[0]:k for k,v in POSITIONS.items() if v[0]}
POSITION_GROUPS = {
    'dl_lb': ['DL','LB'],
    'secondary': ['DB'],
    'wr_cb': ['WR','DB'],
}

# ---------- Field mask ----------
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

def draw_field_polygon_interactive(frame):
    pts=[]; clone=frame.copy()
    def on_click(e,x,y,flags,p):
        if e==cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y)); cv2.circle(clone,(x,y),4,(0,255,255),-1)
            if len(pts)>1: cv2.line(clone,pts[-2],pts[-1],(0,255,255),2)
            cv2.imshow("Draw field - click corners, C to close", clone)
    cv2.imshow("Draw field - click corners, C to close", clone)
    cv2.setMouseCallback("Draw field - click corners, C to close", on_click)
    while True:
        cv2.imshow("Draw field - click corners, C to close", clone)
        if cv2.waitKey(20) & 0xFF == ord('c') and len(pts)>=3: break
    cv2.destroyWindow("Draw field - click corners, C to close")
    mask=np.zeros(frame.shape[:2],dtype=np.uint8)
    cv2.fillPoly(mask,[np.array(pts,np.int32)],255)
    return mask, pts

# ---------- OCR ----------
class JerseyOCR:
    def __init__(self, enabled=True):
        self.enabled = enabled and OCR_AVAILABLE
        self.reader = None
        if self.enabled:
            try: self.reader = easyocr.Reader(['en'], gpu=True)
            except: self.reader = easyocr.Reader(['en'], gpu=False)
    def read(self, crop_bgr):
        if not self.enabled or self.reader is None: return None
        h,w = crop_bgr.shape[:2]
        # jersey is usually upper torso
        roi = crop_bgr[int(h*0.15):int(h*0.6), :]
        if roi.size == 0: return None
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        try:
            res = self.reader.readtext(roi, allowlist='0123456789', detail=0)
            nums = ''.join(ch for s in res for ch in s if ch.isdigit())
            if nums: return nums[:2]
        except: pass
        return None

# ---------- Selection UI ----------
class Selector:
    def __init__(self, frame, detections):
        self.frame = frame.copy()
        self.dets = detections
        self.included = [True]*len(detections)
        self.team = [0]*len(detections)
        self.pos = [None]*len(detections)  # 'DL','LB','DB','OL','WR','QB','RB'
        self.win = "Select - Click toggle | R/B team | 1-7 position | SPACE go"

    def point_in_box(self, x,y, box):
        x1,y1,x2,y2 = box[:4]; return x1<=x<=x2 and y1<=y<=y2

    def draw(self):
        vis = self.frame.copy()
        for i,d in enumerate(self.dets):
            x1,y1,x2,y2 = map(int, d[:4])
            inc = self.included[i]
            team_off = self.team[i]==0
            color = (0,0,255) if team_off else (255,100,0)
            if not inc: color = (120,120,120)
            cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
            pos_str = self.pos[i] or "-"
            label = f"{'OFF' if team_off else 'DEF'} {pos_str} {i}{' -OUT' if not inc else ''}"
            cv2.putText(vis,label,(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
        info = f"Incl {sum(self.included)}/{len(self.dets)}  1=DL 2=LB 3=DB 4=OL 5=WR 6=QB 7=RB 0=clear | R/B team"
        cv2.putText(vis,info,(12,26),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(vis,info,(12,26),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1,cv2.LINE_AA)
        return vis

    def run(self):
        hover_idx = [-1]
        def mouse(e,x,y,flags,p):
            if e==cv2.EVENT_MOUSEMOVE:
                hover_idx[0] = next((i for i,d in enumerate(self.dets) if self.point_in_box(x,y,d)), -1)
            elif e==cv2.EVENT_LBUTTONDOWN:
                for i,d in enumerate(self.dets):
                    if self.point_in_box(x,y,d):
                        if flags & cv2.EVENT_FLAG_SHIFTKEY: self.team[i] ^= 1
                        else: self.included[i] = not self.included[i]
                        break
        cv2.namedWindow(self.win); cv2.setMouseCallback(self.win, mouse)
        while True:
            cv2.imshow(self.win, self.draw())
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'): cv2.destroyWindow(self.win); return None
            if k == ord(' '): break
            if k == ord('r'):
                for i,inc in enumerate(self.included):
                    if inc: self.team[i]=0
            if k == ord('b'):
                for i,inc in enumerate(self.included):
                    if inc: self.team[i]=1
            ch = chr(k) if 32 <= k < 127 else ''
            if ch in POSITIONS:
                pos_name,_ = POSITIONS[ch]
                idx = hover_idx[0]
                if idx >= 0:
                    self.pos[idx] = pos_name
                else:  # apply to all included with no pos yet
                    for i,inc in enumerate(self.included):
                        if inc and self.pos[i] is None: self.pos[i] = pos_name
        cv2.destroyWindow(self.win)
        kept_boxes  = [self.dets[i] for i,inc in enumerate(self.included) if inc]
        kept_teams  = [self.team[i] for i,inc in enumerate(self.included) if inc]
        kept_pos    = [self.pos[i]  for i,inc in enumerate(self.included) if inc]
        return kept_boxes, kept_teams, kept_pos

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--tracker", default="ocsort.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--draw_field", action="store_true")
    ap.add_argument("--field_json", default=None)
    ap.add_argument("--ocr", action="store_true", help="enable jersey OCR (easyocr, slow)")
    ap.add_argument("--ball", action="store_true", default=True, help="track ball (COCO class 32)")
    ap.add_argument("--no_ball", action="store_true")
    ap.add_argument("--px_per_yard", type=float, default=0.0, help="pixels per yard for speed/distance, 0=px only. Calibrate: measure 10yd line in pixels.")
    ap.add_argument("--csv", default="tracks.csv", help="export per-frame CSV")
    args = ap.parse_args()
    track_ball = args.ball and not args.no_ball

    cap = cv2.VideoCapture(args.video)
    ok, first_frame = cap.read()
    if not ok: print("Can't open video"); return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.field_json and os.path.exists(args.field_json):
        poly=json.load(open(args.field_json)); mask=np.zeros(first_frame.shape[:2],dtype=np.uint8)
        cv2.fillPoly(mask,[np.array(poly,np.int32)],255)
    elif args.draw_field:
        mask,poly = draw_field_polygon_interactive(first_frame)
        if args.field_json: json.dump(poly, open(args.field_json,"w"))
    else:
        mask = estimate_field_mask(first_frame)

    model = YOLO(args.model)
    ocr = JerseyOCR(enabled=args.ocr)

    res = model.predict(first_frame, classes=[0], conf=args.conf, iou=args.iou_thresh, imgsz=args.imgsz, verbose=False)[0]
    dets=[]; h,w = first_frame.shape[:2]
    for b in res.boxes:
        x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
        cx=int((x1+x2)/2); cy=int((y1+y2)/2)
        if 0<=cy<h and 0<=cx<w and mask[cy,cx]:
            dets.append([x1,y1,x2,y2,float(b.conf[0])])
    dets = sorted(dets, key=lambda d:(d[2]-d[0])*(d[3]-d[1]), reverse=True)[:30]
    dets = sorted(dets, key=lambda d:d[0])
    if not dets: print("No players found"); return

    selector = Selector(first_frame, dets)
    sel = selector.run()
    if sel is None: return
    kept_boxes, kept_teams, kept_pos = sel
    print(f"Tracking {len(kept_boxes)} players")

    def box_center(b): return np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])
    init_centers = [box_center(b) for b in kept_boxes]
    init_teams   = kept_teams
    init_pos     = kept_pos

    assigned_track_teams = {}
    assigned_track_pos = {}
    track_jersey = {}
    track_trails = defaultdict(lambda: deque(maxlen=30))
    track_speed = {}  # tid -> px/s

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_w,out_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_tracked.mp4", fourcc, fps, (out_w,out_h))

    # distance measure mode
    measure_clicks = []
    measure_pair = []  # [tid1, tid2]
    current_frame_tracks = {}  # tid -> (cx,cy,box)

    def mouse_measure(e,x,y,flags,p):
        if e==cv2.EVENT_LBUTTONDOWN and measure_clicks is not None:
            measure_clicks.append((x,y))

    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", mouse_measure)

    filter_mode = "all"  # all / DL / LB / DB / OL / WR / QB / RB / dl_lb / secondary / wr_cb
    pos_filter_map = {'1':'DL','2':'LB','3':'DB','4':'OL','5':'WR','6':'QB','7':'RB'}

    csv_f = open(args.csv,"w",newline=''); csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame","track_id","team","position","jersey","x","y","speed_px_s","speed_yd_s"])

    frame_idx=0
    while True:
        ok,frame = cap.read()
        if not ok: break

        # player track
        res = model.track(frame, classes=[0], conf=args.conf, iou=args.iou_thresh,
                          persist=True, tracker=args.tracker, imgsz=args.imgsz, verbose=False)[0]

        current_frame_tracks.clear()
        if res.boxes is not None and res.boxes.id is not None:
            xyxys = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            for xyxy,tid in zip(xyxys,ids):
                x1,y1,x2,y2 = map(int, xyxy)
                cx=(x1+x2)//2; cy=(y1+y2)//2
                if not (0<=cy<mask.shape[0] and 0<=cx<mask.shape[1] and mask[cy,cx]): continue
                current_frame_tracks[tid] = (cx,cy,(x1,y1,x2,y2))

                # first-sight team/pos assignment
                if tid not in assigned_track_teams:
                    ctr=np.array([cx,cy])
                    if init_centers:
                        dists=[np.linalg.norm(ctr-ic) for ic in init_centers]
                        k=int(np.argmin(dists))
                        assigned_track_teams[tid]=init_teams[k]
                        assigned_track_pos[tid]=init_pos[k]
                    else:
                        assigned_track_teams[tid]=0; assigned_track_pos[tid]=None

                # speed
                trail = track_trails[tid]; trail.append((frame_idx,cx,cy))
                speed_px_s = 0.0
                if len(trail) >= 2:
                    t0, x0, y0 = trail[0]; t1, x1_, y1_ = trail[-1]
                    dt = max((t1-t0)/fps, 1e-3)
                    dist = np.hypot(x1_-x0, y1_-y0)
                    speed_px_s = dist/dt
                track_speed[tid]=speed_px_s

                # jersey OCR (throttled)
                if ocr.enabled and tid not in track_jersey and frame_idx % 15 == 0:
                    crop = frame[max(0,y1):min(out_h,y2), max(0,x1):min(out_w,x2)]
                    if crop.size>0:
                        num = ocr.read(crop)
                        if num: track_jersey[tid]=num

        # ball track
        ball_xy = None
        if track_ball:
            bres = model.predict(frame, classes=[32], conf=0.15, imgsz=args.imgsz, verbose=False)[0]
            best=None; best_conf=0
            for b in bres.boxes:
                x1,y1,x2,y2 = map(int,b.xyxy[0].cpu().numpy())
                cx=(x1+x2)//2; cy=(y1+y2)//2
                if 0<=cy<mask.shape[0] and 0<=cx<mask.shape[1] and mask[cy,cx]==0: continue
                conf=float(b.conf[0])
                if conf>best_conf: best_conf=conf; best=(x1,y1,x2,y2,cx,cy)
            if best: ball_xy = best[4:]

        # distance measure clicks -> map to nearest track
        if measure_clicks:
            for mx,my in measure_clicks:
                nearest=None; nd=1e9
                for tid,(cx,cy,_) in current_frame_tracks.items():
                    d=np.hypot(cx-mx,cy-my)
                    if d<nd and d<60: nd=d; nearest=tid
                if nearest is not None:
                    if len(measure_pair)<2 and nearest not in measure_pair:
                        measure_pair.append(nearest)
            measure_clicks.clear()
            if len(measure_pair)>2: measure_pair = measure_pair[-2:]

        # draw
        def pos_allowed(pos):
            if filter_mode=="all": return True
            if filter_mode in pos_filter_map.values(): return pos==filter_mode
            if filter_mode=="dl_lb": return pos in ["DL","LB"]
            if filter_mode=="secondary": return pos=="DB"
            if filter_mode=="wr_cb": return pos in ["WR","DB"]
            return True

        for tid,(cx,cy,(x1,y1,x2,y2)) in current_frame_tracks.items():
            team = assigned_track_teams.get(tid,0)
            pos = assigned_track_pos.get(tid)
            if not pos_allowed(pos): continue
            color = (0,0,255) if team==0 else (255,100,0)
            # trail
            pts = [(int(x),int(y)) for _,x,y in track_trails[tid]]
            for i in range(1,len(pts)): cv2.line(frame,pts[i-1],pts[i],color,1)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            jersey = track_jersey.get(tid,"")
            speed_px = track_speed.get(tid,0)
            speed_yd = speed_px/args.px_per_yard if args.px_per_yard>0 else 0
            label = f"{'OFF' if team==0 else 'DEF'} {pos or ''} #{jersey or tid}"
            if speed_px>0: label += f" {speed_yd:.1f}yd/s" if args.px_per_yard>0 else f" {speed_px:.0f}px/s"
            cv2.putText(frame,label,(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)

            csv_w.writerow([frame_idx,tid, "OFF" if team==0 else "DEF", pos or "", jersey or "",
                            cx,cy, f"{speed_px:.2f}", f"{speed_yd:.2f}" if args.px_per_yard>0 else ""])

        # ball
        if ball_xy:
            bx,by = ball_xy
            cv2.circle(frame,(bx,by),8,(0,255,255),2)
            cv2.putText(frame,"BALL",(bx+10,by),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,cv2.LINE_AA)

        # measure line
        if len(measure_pair)==2:
            t1,t2 = measure_pair
            if t1 in current_frame_tracks and t2 in current_frame_tracks:
                x1,y1,_ = current_frame_tracks[t1]
                x2,y2,_ = current_frame_tracks[t2]
                cv2.line(frame,(x1,y1),(x2,y2),(255,255,255),2)
                px_dist = np.hypot(x2-x1,y2-y1)
                if args.px_per_yard>0:
                    dist_str = f"{px_dist/args.px_per_yard:.1f} yd"
                else:
                    dist_str = f"{px_dist:.0f} px"
                cv2.putText(frame, dist_str, ((x1+x2)//2, (y1+y2)//2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)

        # HUD
        hud = f"Filter:{filter_mode} | D=measure click 2 players | C=clear measure | 1-7 filter | 0 all | G DL+LB | H secondary | J WR/CB"
        cv2.putText(frame,hud,(12,26),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame,hud,(12,26),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        if measure_pair:
            mp = f"Measure: {' + '.join(map(str,measure_pair))}"
            cv2.putText(frame,mp,(12,48),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)

        out.write(frame)
        cv2.imshow("Tracking", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord('d'): measure_pair.clear()
        if k == ord('c'): measure_pair.clear()
        if k == ord('0'): filter_mode="all"
        if chr(k) in pos_filter_map if 0<=k<256 else False:
            filter_mode = pos_filter_map[chr(k)]
        if k == ord('g'): filter_mode="dl_lb"
        if k == ord('h'): filter_mode="secondary"
        if k == ord('j'): filter_mode="wr_cb"

        frame_idx+=1

    csv_f.close()
    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"Saved output_tracked.mp4  tracks -> {args.csv}")

if __name__ == "__main__":
    main()
