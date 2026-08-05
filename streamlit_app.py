import streamlit as st
import cv2, numpy as np, tempfile, os
from collections import defaultdict, deque
from ultralytics import YOLO
import pandas as pd

st.set_page_config(page_title="Football Tracker", layout="wide")
st.title("🏈 Football 22-Player Tracker")

# ---------- OCR (optional) ----------
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

POSITIONS = ["DL","LB","DB","OL","WR","QB","RB"]

# Roboflow American Football model class map
# football-players-zm06l classes: Center, QB, db, lb, skill
RF_FB_CLASS_MAP = {
    "center": "OL",
    "qb": "QB",
    "db": "DB",
    "lb": "LB",
    "skill": "WR",   # skill = RB/FB/TE/WR – default to WR, user can override
}

def estimate_field_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return np.ones(frame.shape[:2], dtype=np.uint8)*255
    c = max(cnts, key=cv2.contourArea)
    out = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(out, [c], -1, 255, -1)
    return out

class JerseyOCR:
    def __init__(self, enabled):
        self.enabled = enabled and OCR_AVAILABLE
        self.reader = None
        if self.enabled:
            try: self.reader = easyocr.Reader(['en'], gpu=False)
            except: self.reader = None
            self.enabled = self.reader is not None
    def read(self, crop):
        if not self.enabled or self.reader is None: return None
        h,_ = crop.shape[:2]
        roi = crop[int(h*0.15):int(h*0.6), :]
        if roi.size == 0: return None
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        try:
            res = self.reader.readtext(roi, allowlist='0123456789', detail=0)
            nums = ''.join(ch for s in res for ch in s if ch.isdigit())
            return nums[:2] if nums else None
        except: return None

def clear_tracker_callbacks(m):
    try:
        if hasattr(m, 'predictor') and m.predictor and hasattr(m.predictor, 'callbacks'):
            m.predictor.callbacks = {k: [] for k in m.predictor.callbacks}
    except Exception:
        pass

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Model")
    model_source = st.radio("Player detector", [
        "Ultralytics COCO (person)",
        "Roboflow American Football",
        "Custom .pt weights"
    ], index=1)

    rf_api_key = None
    rf_model_id = "football-players-zm06l/8"
    custom_weights_path = None

    if model_source == "Ultralytics COCO (person)":
        model_name = st.selectbox("YOLO model", ["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt"], index=2)
        player_class_ids = [0]
        auto_pos_map = {}
        weights_path = model_name
    elif model_source == "Roboflow American Football":
        st.caption("Model: bronkscottema/football-players-zm06l – American football, classes: Center, QB, DB, LB, skill")
        try:
            import roboflow
            RF_AVAILABLE = True
        except ImportError:
            RF_AVAILABLE = False
            st.warning("pip install roboflow  (then restart)")
        rf_api_key = st.text_input("Roboflow API key", type="password",
            help="Get a free key at app.roboflow.com – used once to download weights, then runs offline")
        rf_model_id = st.text_input("Model ID", value="football-players-zm06l/8")
        if rf_api_key and RF_AVAILABLE:
            # download weights if needed
            import pathlib
            cache_dir = pathlib.Path(tempfile.gettempdir()) / "rf_football_models" / rf_model_id.replace("/","_")
            weights_path = cache_dir / "weights" / "best.pt"
            if not weights_path.exists():
                with st.spinner("Downloading Roboflow model weights… first run only"):
                    try:
                        from roboflow import Roboflow
                        rf = Roboflow(api_key=rf_api_key)
                        ws, proj, ver = rf_model_id.split("/") if "/" in rf_model_id and rf_model_id.count("/") == 2 else ("bronkscottema", "football-players-zm06l", "8")
                        if rf_model_id.count("/") == 1:
                            proj, ver = rf_model_id.split("/")
                            ws = "bronkscottema"
                        project = rf.workspace(ws).project(proj)
                        dataset = project.version(int(ver)).download("yolov8", location=str(cache_dir))
                        # weights should be at cache_dir / "weights" / "best.pt" – actually dataset download gives training data, not weights
                        # fallback: try project.version().deploy("yolov8")
                    except Exception as e:
                        st.error(f"Roboflow download failed: {e}")
                        st.info("Alternatively: download weights manually from Roboflow Universe → Download → YOLOv8 → \"Download Model Weights (best.pt)\", then use 'Custom .pt weights' option.")
                        st.stop()
            if not weights_path.exists():
                # try common alternate locations
                for p in list(cache_dir.rglob("best.pt")):
                    weights_path = p; break
            if not weights_path.exists():
                st.error(f"Weights not found at {weights_path}. Download best.pt from https://universe.roboflow.com/bronkscottema/football-players-zm06l and use 'Custom .pt weights' option.")
                st.stop()
            weights_path = str(weights_path)
            # load model to get class names
            _tmp_model = YOLO(weights_path)
            names = _tmp_model.names  # {0: 'Center', 1: 'QB', ...}
            player_class_ids = list(names.keys())
            auto_pos_map = {cid: RF_FB_CLASS_MAP.get(names[cid].lower(), None) for cid in player_class_ids}
            model_name = weights_path
            st.success(f"Loaded {weights_path} – classes: {', '.join(names.values())}")
        else:
            st.info("Enter your Roboflow API key to auto-download, or use 'Custom .pt weights' if you've already downloaded best.pt")
            st.stop()
    else:  # Custom .pt
        up = st.file_uploader("Upload best.pt", type=["pt"])
        if up:
            p = os.path.join(tempfile.gettempdir(), up.name)
            with open(p, "wb") as f: f.write(up.read())
            custom_weights_path = p
            weights_path = p
            _tmp_model = YOLO(weights_path)
            names = _tmp_model.names
            player_class_ids = list(names.keys())
            # try to map known football classes, else no auto-pos
            auto_pos_map = {cid: RF_FB_CLASS_MAP.get(str(names[cid]).lower(), None) for cid in player_class_ids}
            model_name = weights_path
            st.success(f"Loaded custom weights – classes: {', '.join(str(v) for v in names.values())}")
        else:
            st.info("Upload a YOLOv8 .pt file. For the Roboflow American Football model: https://universe.roboflow.com/bronkscottema/football-players-zm06l → Download → Model weights (best.pt)")
            st.stop()

    st.divider()
    st.header("Config")
    conf = st.slider("Player detection conf", 0.05, 0.6, 0.15, 0.05)
    imgsz = st.selectbox("imgsz", [640,960,1280], index=1)
    use_field_mask = st.checkbox("Field mask (ignore sidelines)", value=False,
        help="Turn off if it's eating your players – the football-specific model is better at ignoring sidelines on its own")
    use_ocr = st.checkbox("Jersey OCR (slow)", value=False, disabled=not OCR_AVAILABLE)
    track_ball = st.checkbox("Track ball (COCO sports ball – weak)", value=False)
    ball_conf = st.slider("Ball conf", 0.05, 0.5, 0.1, 0.05, disabled=not track_ball)
    px_per_yard = st.number_input("px per yard (0 = px only)", min_value=0.0, value=0.0, step=0.1)
    st.caption("Tip: calibrate px_per_yard for real yd/s speeds")

uploaded = st.file_uploader("Upload play video", type=["mp4","mov","avi","m4v"])
if not uploaded:
    st.info("Upload a football play. The Roboflow American Football model should find ~15-22 players vs ~4-10 with COCO.")
    st.stop()

tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded.read()); tfile.close()
video_path = tfile.name

cap = cv2.VideoCapture(video_path)
ok, first_frame = cap.read()
cap.release()
if not ok:
    st.error("Can't open video"); st.stop()

h,w = first_frame.shape[:2]
field_mask = estimate_field_mask(first_frame) if use_field_mask else np.ones((h,w), dtype=np.uint8)*255

# ---------- First-frame player picker ----------
@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

model = load_model(weights_path)
clear_tracker_callbacks(model)

# get class names for auto-position tagging
model_class_names = model.names if hasattr(model, 'names') else {}

res = model.predict(first_frame, classes=player_class_ids, conf=conf, imgsz=imgsz, verbose=False)[0]
dets = []
for b in res.boxes:
    x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
    cx=int((x1+x2)/2); cy=int((y1+y2)/2)
    if 0<=cy<h and 0<=cx<w and field_mask[cy,cx]:
        cls_id = int(b.cls[0].cpu().numpy()) if hasattr(b, 'cls') else 0
        dets.append([float(x1),float(y1),float(x2),float(y2), float(b.conf[0]), cls_id])

# keep up to 30 largest
dets = sorted(dets, key=lambda d:(d[2]-d[0])*(d[3]-d[1]), reverse=True)[:30]
dets = sorted(dets, key=lambda d:d[0])

st.subheader(f"1️⃣ Select players  —  detected {len(dets)} in-frame")
if len(dets) < 10:
    st.warning(f"Only {len(dets)} players detected. Try: lower conf, larger imgsz, or turn off Field mask.")
else:
    st.success(f"Found {len(dets)} players – much better than COCO!")
st.caption("Offense = Red, Defense = Blue · Position auto-tagged from model if available")

def guess_team(box):
    cx = (box[0]+box[2])/2
    return "OFF" if cx < w/2 else "DEF"

# Select All / Deselect All
c1, c2, c3 = st.columns([1,1,6])
with c1:
    if st.button("Select all"):
        for i in range(len(dets)): st.session_state[f"inc_{i}"] = True
        st.rerun()
with c2:
    if st.button("Deselect all"):
        for i in range(len(dets)): st.session_state[f"inc_{i}"] = False
        st.rerun()

cols = st.columns(4)
player_meta = {}
for i, d in enumerate(dets):
    x1,y1,x2,y2,conf_score,cls_id = d if len(d) == 6 else (*d, 0)
    crop = first_frame[int(y1):int(y2), int(x1):int(x2)]
    # auto-position from model class
    auto_pos = auto_pos_map.get(cls_id, None)
    auto_label = model_class_names.get(cls_id, "") if model_class_names else ""
    with cols[i % 4]:
        if crop.size > 0:
            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width=120)
        if auto_label:
            st.caption(f"🤖 {auto_label} → {auto_pos or '—'} · {conf_score:.2f}")
        inc_key = f"inc_{i}"
        if inc_key not in st.session_state: st.session_state[inc_key] = True
        inc = st.checkbox(f"Include #{i}", key=inc_key)
        team = st.radio(f"Team {i}", ["OFF","DEF"], index=0 if guess_team(d)=="OFF" else 1,
                        horizontal=True, key=f"team_{i}", label_visibility="collapsed")
        # pre-select position from model
        pos_options = ["—"]+POSITIONS
        pos_default_idx = pos_options.index(auto_pos) if auto_pos in pos_options else 0
        pos = st.selectbox(f"Pos {i}", pos_options, index=pos_default_idx, key=f"pos_{i}", label_visibility="collapsed")
        player_meta[i] = {"inc": inc, "team": 0 if team=="OFF" else 1,
                          "pos": None if pos=="—" else pos, "box": d[:4], "cls": cls_id}

kept = [(i, m) for i,m in player_meta.items() if m["inc"]]
st.write(f"Tracking **{len(kept)} players**")
if len(kept) == 0:
    st.stop()

init_centers = [np.array([ (m["box"][0]+m["box"][2])/2, (m["box"][1]+m["box"][3])/2 ]) for _,m in kept]
init_teams   = [m["team"] for _,m in kept]
init_pos     = [m["pos"]  for _,m in kept]

# ---------- Run tracker ----------
if st.button("▶️ Run tracking", type="primary"):
    ocr = JerseyOCR(use_ocr)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (vw, vh))

    assigned_team = {}; assigned_pos = {}; jersey_map = {}
    allowed_tids = set()
    trails = defaultdict(lambda: deque(maxlen=30))
    speed_map = {}
    rows = []

    progress = st.progress(0, "Tracking…")
    frame_placeholder = st.empty()
    frame_idx = 0

    # clear any tracker callbacks before tracking run
    clear_tracker_callbacks(model)

    # distance gate for initial track-to-selection matching (pixels)
    match_gate = max(vw, vh) * 0.15

    # ball detector – always use COCO yolov8n for ball, separate from player model
    ball_model = None
    if track_ball:
        try:
            ball_model = YOLO("yolov8n.pt")
        except Exception:
            ball_model = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.track(frame, classes=player_class_ids, conf=conf, persist=True,
                          tracker="ocsort.yaml", imgsz=imgsz, verbose=False)[0]
        current_tracks = {}
        if res.boxes is not None and res.boxes.id is not None:
            xyxys = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else [0]*len(ids)
            for xyxy, tid, cls_id in zip(xyxys, ids, clss):
                x1,y1,x2,y2 = map(int, xyxy)
                cx=(x1+x2)//2; cy=(y1+y2)//2
                if use_field_mask and not (0<=cy<field_mask.shape[0] and 0<=cx<field_mask.shape[1] and field_mask[cy,cx]): 
                    continue
                current_tracks[tid] = (cx,cy,(x1,y1,x2,y2))
                # first-sight assignment to nearest selected player, with gate
                if tid not in assigned_team:
                    ctr = np.array([cx,cy])
                    if init_centers:
                        dists = [np.linalg.norm(ctr-ic) for ic in init_centers]
                        k = int(np.argmin(dists))
                        if dists[k] <= match_gate:
                            assigned_team[tid] = init_teams[k]
                            assigned_pos[tid]  = init_pos[k]
                            allowed_tids.add(tid)
                        else:
                            continue
                    else:
                        assigned_team[tid]=0; assigned_pos[tid]=None
                        allowed_tids.add(tid)
                if tid not in allowed_tids:
                    continue
                tr = trails[tid]; tr.append((frame_idx,cx,cy))
                speed = 0.0
                if len(tr) >= 2:
                    t0,x0,y0 = tr[0]; t1,x1_,y1_ = tr[-1]
                    speed = np.hypot(x1_-x0, y1_-y0) / max((t1-t0)/fps, 1e-3)
                speed_map[tid] = speed
                if ocr.enabled and tid not in jersey_map and frame_idx % 15 == 0:
                    crop = frame[max(0,y1):min(vh,y2), max(0,x1):min(vw,x2)]
                    if crop.size>0:
                        num = ocr.read(crop)
                        if num: jersey_map[tid] = num

        # ball – run every 3 frames, use separate COCO model
        ball_xy = None
        if track_ball and ball_model and frame_idx % 3 == 0:
            bres = ball_model.predict(frame, classes=[32], conf=ball_conf, imgsz=imgsz, verbose=False)[0]
            best_conf = 0
            for b in bres.boxes:
                bx1,by1,bx2,by2 = map(int, b.xyxy[0].cpu().numpy())
                bcx=(bx1+bx2)//2; bcy=(by1+by2)//2
                if use_field_mask and 0<=bcy<field_mask.shape[0] and 0<=bcx<field_mask.shape[1] and field_mask[bcy,bcx]==0: continue
                c = float(b.conf[0])
                if c > best_conf: best_conf=c; ball_xy=(bcx,bcy,bx1,by1,bx2,by2)

        # draw – only allowed_tids
        vis = frame.copy()
        for tid,(cx,cy,(x1,y1,x2,y2)) in current_tracks.items():
            if tid not in allowed_tids: continue
            team = assigned_team.get(tid,0)
            pos = assigned_pos.get(tid)
            color = (0,0,255) if team==0 else (255,100,0)
            pts = [(int(x),int(y)) for _,x,y in trails[tid]]
            for j in range(1,len(pts)): cv2.line(vis,pts[j-1],pts[j],color,1)
            cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
            jersey = jersey_map.get(tid,"")
            spd = speed_map.get(tid,0)
            spd_yd = spd/px_per_yard if px_per_yard>0 else 0
            label = f"{'OFF' if team==0 else 'DEF'} {pos or ''} #{jersey or tid}"
            if spd>0: label += f" {spd_yd:.1f}yd/s" if px_per_yard>0 else f" {spd:.0f}px/s"
            cv2.putText(vis,label,(x1,max(12,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
            rows.append([frame_idx,tid,"OFF" if team==0 else "DEF", pos or "", jersey or "", cx,cy, spd, spd_yd])

        if ball_xy:
            bcx,bcy,_,_,_,_ = ball_xy
            cv2.circle(vis,(bcx,bcy),8,(0,255,255),2)
            cv2.putText(vis,"BALL",(bcx+10,bcy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,cv2.LINE_AA)

        out.write(vis)
        if frame_idx % max(1,int(fps/4)) == 0:
            frame_placeholder.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if total_frames: progress.progress(min(frame_idx/total_frames,1.0))
        frame_idx += 1

    cap.release(); out.release()
    progress.empty()
    st.success(f"Done – {frame_idx} frames tracked, {len(allowed_tids)} locked IDs")
    st.video(out_path)

    df = pd.DataFrame(rows, columns=["frame","track_id","team","position","jersey","x","y","speed_px_s","speed_yd_s"])
    st.session_state["tracks_df"] = df
    st.session_state["video_out"] = out_path
    with open(out_path,"rb") as f: st.download_button("⬇️ Download tracked MP4", f, "tracked.mp4", "video/mp4")
    st.download_button("⬇️ Download tracks CSV", df.to_csv(index=False).encode(), "tracks.csv", "text/csv")

# ---------- Analysis ----------
if "tracks_df" in st.session_state:
    df = st.session_state["tracks_df"]
    st.divider()
    st.subheader("2️⃣ Analyze")
    col1,col2,col3 = st.columns(3)
    with col1:
        pos_filter = st.multiselect("Position filter", POSITIONS, default=[])
    with col2:
        team_filter = st.multiselect("Team", ["OFF","DEF"], default=["OFF","DEF"])
    with col3:
        combo = st.selectbox("Quick combo", ["—","DL+LB","Secondary (DB)","WR vs CB"], index=0)
    if combo == "DL+LB": pos_filter = ["DL","LB"]
    elif combo == "Secondary (DB)": pos_filter = ["DB"]
    elif combo == "WR vs CB": pos_filter = ["WR","DB"]
    fdf = df[df["team"].isin(team_filter)]
    if pos_filter: fdf = fdf[fdf["position"].isin(pos_filter)]
    st.dataframe(fdf.head(200), use_container_width=True)
    st.markdown("**Measure distance**")
    tids = sorted(df["track_id"].unique())
    c1,c2 = st.columns(2)
    with c1: t1 = st.selectbox("Player A", tids, index=0 if tids else None, key="meas_a")
    with c2: t2 = st.selectbox("Player B", tids, index=min(1,len(tids)-1) if len(tids)>1 else 0, key="meas_b")
    if tids and t1 is not None and t2 is not None and t1 != t2:
        merged = df[df["track_id"]==t1][["frame","x","y"]].merge(
                 df[df["track_id"]==t2][["frame","x","y"]], on="frame", suffixes=("_a","_b"))
        if not merged.empty:
            last = merged.iloc[-1]
            px_dist = np.hypot(last.x_a-last.x_b, last.y_a-last.y_b)
            if px_per_yard > 0:
                st.metric("Distance", f"{px_dist/px_per_yard:.1f} yd", f"{px_dist:.0f} px")
            else:
                st.metric("Distance", f"{px_dist:.0f} px")
    if not fdf.empty:
        st.line_chart(fdf.groupby("frame")["speed_yd_s" if px_per_yard>0 else "speed_px_s"].mean(), height=200)
