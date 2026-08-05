# streamlit_app_v3.py
# Football Tracker v3 – Roboflow Workflow + metrics UI
# M4 MacBook Air optimized
#
# Features:
#  - Roboflow Workflow: brady-powell / football-player-and-ball-tracker-1785894897417
#  - Player multi-select: Select All / Deselect All / Position filter (DL/LB/DB/OL/WR/QB/RB)
#  - Manual box add
#  - Matchup mode: 2 players guarding each other → separation chart
#  - Speed (mph), Acceleration (50/100 frame windows), Distance ran, Displacement from snap
#  - Player-to-player distance (1 vs 1 / 1 vs many)
#  - Time-segment A/B analysis with averages
#  - Scrubber with forward/reverse frame stepping
#  - CSV export

import streamlit as st
import cv2, numpy as np, tempfile, os, json
import pandas as pd
from collections import defaultdict

st.set_page_config(page_title="Football Tracker v3", layout="wide")
st.title("🏈 Football Tracker v3 – Roboflow")

try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

from metrics import TrackMetrics
try:
    from roboflow_detector import run_workflow_frame, parse_workflow_detections, ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID
    RF_AVAILABLE = True
except Exception as e:
    RF_AVAILABLE = False
    RF_IMPORT_ERROR = str(e)

POSITIONS = ["DL","LB","DB","OL","WR","QB","RB"]

# ---------- Sidebar: Model ----------
with st.sidebar:
    st.header("Roboflow Workflow")
    if RF_AVAILABLE:
        st.success(f"{ROBOFLOW_WORKSPACE} / {ROBOFLOW_WORKFLOW_ID}")
    else:
        st.error(f"Roboflow SDK missing: {RF_IMPORT_ERROR if 'RF_IMPORT_ERROR' in globals() else ''}")
        st.code("pip install inference python-dotenv", language="bash")
        st.stop()
    conf_thresh = st.slider("Min conf (post-filter)", 0.05, 0.8, 0.25, 0.05)
    px_per_yard = st.number_input("px / yard (0 = px only)", 0.0, 100.0, 0.0, 0.1,
        help="Calibrate for mph. Measure a 10-yd line in pixels.")
    accel_window = st.selectbox("Acceleration window", [30,50,100,150], index=1,
        help="Δv over N frames")
    st.divider()
    st.caption("v3 • metrics engine: metrics.py")

uploaded = st.file_uploader("Upload play video", type=["mp4","mov","avi","m4v","mkv"])
if not uploaded:
    st.info("Upload a football clip to start. The Workflow runs server-side, tracking runs locally on your M4.")
    st.stop()

# save upload
tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded.read()); tfile.close()
video_path = tfile.name

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

st.write(f"{w}×{h} @ {fps:.1f} fps, {n_frames} frames")

# ---------- Run detection + tracking ----------
# Cache by video file hash so re-selecting players doesn't re-run RF
import hashlib
with open(video_path, "rb") as f:
    vid_hash = hashlib.md5(f.read()).hexdigest()[:12]

cache_path = os.path.join(tempfile.gettempdir(), f"ft_v3_{vid_hash}_tracks.json")

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    au=(a[2]-a[0])*(a[3]-a[1]); bu=(b[2]-b[0])*(b[3]-b[1])
    return inter/(au+bu-inter+1e-6)

class SimpleIoUTracker:
    def __init__(self, iou_thresh=0.3, max_age=30):
        self.iou_thresh=iou_thresh; self.max_age=max_age; self.next_id=1; self.tracks={}
    def update(self, dets):
        if not self.tracks:
            out=[]; 
            for d in dets:
                tid=self.next_id; self.next_id+=1
                self.tracks[tid]={"box":d[:4],"age":0}
                out.append((*d[:4], tid))
            return out
        tids=list(self.tracks.keys()); tboxes=[self.tracks[i]["box"] for i in tids]
        matches={}; used=set()
        for ti,tb in enumerate(tboxes):
            best_j,best_i=-1,self.iou_thresh
            for dj,det in enumerate(dets):
                if dj in used: continue
                v=iou(tb,det[:4])
                if v>best_i: best_i,best_j=v,dj
            if best_j>=0: matches[tids[ti]]=best_j; used.add(best_j)
        out=[]
        for tid,dj in matches.items():
            box=dets[dj][:4]; self.tracks[tid]={"box":box,"age":0}; out.append((*box,tid))
        for dj,det in enumerate(dets):
            if dj in used: continue
            tid=self.next_id; self.next_id+=1
            self.tracks[tid]={"box":det[:4],"age":0}; out.append((*det[:4],tid))
        for tid in list(self.tracks.keys()):
            if tid not in [o[4] for o in out]:
                self.tracks[tid]["age"]+=1
            if self.tracks[tid]["age"]>self.max_age: del self.tracks[tid]
        return out

if os.path.exists(cache_path):
    st.info(f"Found cached tracks: {cache_path}")
    use_cache = st.checkbox("Use cached tracks", value=True)
else:
    use_cache = False

if use_cache:
    with open(cache_path) as f:
        cached = json.load(f)
    all_tracks = cached["tracks"]  # {frame: [[x1,y1,x2,y2,tid], ...]}
    metrics_blob = cached.get("metrics_history", {})
else:
    # run full video through RF + tracker
    cap = cv2.VideoCapture(video_path)
    tracker = SimpleIoUTracker(iou_thresh=0.3, max_age=int(fps))
    metrics = TrackMetrics(fps=fps, px_per_yard=px_per_yard)
    all_tracks = {}
    progress = st.progress(0, text="Running Roboflow Workflow…")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        wf = run_workflow_frame(frame)
        players, balls = parse_workflow_detections(wf)
        # filter by conf
        players = [p for p in players if p[4] >= conf_thresh]
        dets = [(x1,y1,x2,y2,conf) for x1,y1,x2,y2,conf,_,_ in players]
        tracks = tracker.update(dets)
        all_tracks[str(frame_idx)] = [[float(x1),float(y1),float(x2),float(y2),int(tid)] for x1,y1,x2,y2,tid in tracks]
        metrics.update(frame_idx, [{"id":tid,"x1":x1,"y1":y1,"x2":x2,"y2":y2} for x1,y1,x2,y2,tid in tracks])
        frame_idx += 1
        if n_frames: progress.progress(min(frame_idx/n_frames,1.0), text=f"Frame {frame_idx}/{n_frames}")
    cap.release()
    progress.empty()
    # save cache
    with open(cache_path, "w") as f:
        json.dump({"tracks": all_tracks, "fps": fps, "px_per_yard": px_per_yard,
                   "metrics_history": metrics.history}, f, default=str)
    # rebuild metrics object from history for UI
    metrics = TrackMetrics(fps=fps, px_per_yard=px_per_yard)
    metrics.history = defaultdict(list, {int(k): v for k,v in metrics_blob.items()} if 'metrics_blob' in globals() else {})
    # actually re-feed
    if not metrics.history:
        # we already have metrics from the run above
        pass

# Rebuild metrics object from cached tracks if needed
if use_cache:
    metrics = TrackMetrics(fps=fps, px_per_yard=px_per_yard)
    for f_str, trks in all_tracks.items():
        fi = int(f_str)
        metrics.update(fi, [{"id":int(t[4]),"x1":t[0],"y1":t[1],"x2":t[2],"y2":t[3]} for t in trks])

# ---------- Player selector ----------
all_tids = sorted({int(t[4]) for trks in all_tracks.values() for t in trks})
if not all_tids:
    st.error("No players tracked – check roboflow_detector.parse_workflow_detections() matches your Workflow output keys.")
    st.stop()

left, right = st.columns([1.2, 1.8])

with left:
    st.subheader("Players")
    c1,c2,c3 = st.columns(3)
    if c1.button("Select all"): st.session_state.sel_tids = set(all_tids)
    if c2.button("Deselect all"): st.session_state.sel_tids = set()
    if "sel_tids" not in st.session_state: st.session_state.sel_tids = set(all_tids[:min(6, len(all_tids))])
    
    # position filter chips (placeholder – assign positions in the multiselect below)
    pos_filter = st.multiselect("Position filter", POSITIONS, default=[])
    st.caption("Position tags are per-track – assign below after selecting players." if not pos_filter else "")

    sel = st.multiselect("Tracked IDs", all_tids, default=sorted(st.session_state.sel_tids), key="ms_tids")
    st.session_state.sel_tids = set(sel)

    # manual box add
    with st.expander("➕ Manual box add"):
        st.caption("If RF missed a player in a frame: enter frame # and box coords, it gets a new track ID.")
        mb_frame = st.number_input("Frame", 0, n_frames-1 if n_frames else 9999, 0)
        mb_x1 = st.number_input("x1", 0, w, 0)
        mb_y1 = st.number_input("y1", 0, h, 0)
        mb_x2 = st.number_input("x2", 0, w, 100)
        mb_y2 = st.number_input("y2", 0, h, 100)
        if st.button("Inject box"):
            new_tid = max(all_tids)+1 if all_tids else 1
            key = str(mb_frame)
            if key not in all_tracks: all_tracks[key] = []
            all_tracks[key].append([mb_x1, mb_y1, mb_x2, mb_y2, new_tid])
            st.success(f"Added TID {new_tid} at frame {mb_frame}. Re-run analysis to propagate.")
    
    st.divider()
    st.subheader("Matchup / Distance")
    matchup_a = st.selectbox("Player A", all_tids, index=0 if all_tids else None)
    matchup_b = st.selectbox("Player B", all_tids, index=min(1, len(all_tids)-1) if all_tids else None)
    vs_many = st.checkbox("A vs all selected", value=False)
    
with right:
    st.subheader("Scrubber")
    max_frame = max((int(k) for k in all_tracks.keys()), default=0)
    scrub_frame = st.slider("Frame", 0, max_frame, 0, key="scrub")
    col_prev, col_next = st.columns(2)
    if col_prev.button("◀ Prev"): st.session_state.scrub = max(0, scrub_frame-1); st.rerun()
    if col_next.button("Next ▶"): st.session_state.scrub = min(max_frame, scrub_frame+1); st.rerun()

    # draw frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, scrub_frame)
    ok, vis = cap.read(); cap.release()
    if ok:
        trks = all_tracks.get(str(scrub_frame), [])
        for x1,y1,x2,y2,tid in trks:
            color = (0,255,0) if tid in st.session_state.sel_tids else (150,150,150)
            cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            cv2.putText(vis, str(tid), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # draw matchup line
        def get_pos(tid, f):
            tr = all_tracks.get(str(f), [])
            for x1,y1,x2,y2,tt in tr:
                if tt == tid: return ((x1+x2)/2, (y1+y2)/2)
            return None
        pa = get_pos(matchup_a, scrub_frame); pb = get_pos(matchup_b, scrub_frame)
        if pa and pb:
            cv2.line(vis, (int(pa[0]),int(pa[1])), (int(pb[0]),int(pb[1])), (0,200,255), 2)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

# ---------- Time segment analysis ----------
st.subheader("Metrics")
seg_a, seg_b = st.columns(2)
with seg_a:
    seg_start = st.number_input("Segment start frame", 0, max_frame, 0)
with seg_b:
    seg_end = st.number_input("Segment end frame", 0, max_frame, max_frame)

sel_tids = sorted(st.session_state.sel_tids) if st.session_state.sel_tids else all_tids[:4]
rows = []
for tid in sel_tids:
    summ = metrics.summary(tid, start_frame=seg_start, end_frame=seg_end, accel_delta=accel_window)
    rows.append({
        "TID": tid,
        f"Avg speed ({summ['units_speed']})": round(summ["avg_speed"],2),
        f"Max speed ({summ['units_speed']})": round(summ["max_speed"],2),
        "Max accel": round(summ["max_accel"],2),
        f"Distance ran ({summ['units_dist']})": round(summ["distance_ran"],2),
        f"Displacement ({summ['units_dist']})": round(summ["displacement"],2),
    })
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- Charts ----------
if sel_tids:
    import altair as alt
    # speed chart
    speed_rows = []
    for tid in sel_tids:
        t, v = metrics.speed_series(tid)
        for tt, vv in zip(t, v):
            if seg_start <= tt <= seg_end:
                speed_rows.append({"frame": tt, "speed": vv, "TID": str(tid)})
    if speed_rows:
        sdf = pd.DataFrame(speed_rows)
        chart = alt.Chart(sdf).mark_line().encode(
            x="frame:Q", y="speed:Q", color="TID:N"
        ).properties(title="Speed over time", height=250)
        st.altair_chart(chart, use_container_width=True)

    # matchup separation chart
    if matchup_a is not None and matchup_b is not None and not vs_many:
        sep_x, sep_y = [], []
        for f_str in sorted(all_tracks.keys(), key=int):
            f = int(f_str)
            if f < seg_start or f > seg_end: continue
            d = metrics.player_distance(matchup_a, matchup_b, f)
            sep_x.append(f); sep_y.append(d)
        if sep_x:
            mdf = pd.DataFrame({"frame": sep_x, "separation": sep_y})
            mc = alt.Chart(mdf).mark_line(color="#ff8800").encode(
                x="frame:Q", y="separation:Q"
            ).properties(title=f"Separation: {matchup_a} vs {matchup_b}", height=200)
            st.altair_chart(mc, use_container_width=True)

    # A vs many distance
    if vs_many and matchup_a is not None:
        many_rows = []
        for tid in sel_tids:
            if tid == matchup_a: continue
            for f in range(seg_start, seg_end+1, max(1, (seg_end-seg_start)//60)):
                d = metrics.player_distance(matchup_a, tid, f)
                many_rows.append({"frame": f, "dist": d, "opponent": str(tid)})
        if many_rows:
            mdf = pd.DataFrame(many_rows)
            mc = alt.Chart(mdf).mark_line().encode(x="frame:Q", y="dist:Q", color="opponent:N")
            st.altair_chart(mc, use_container_width=True)

# ---------- Export ----------
if st.button("Export metrics CSV"):
    df.to_csv("/tmp/ft_v3_metrics.csv", index=False)
    st.success("Saved /tmp/ft_v3_metrics.csv")
    st.download_button("Download CSV", df.to_csv(index=False), "football_metrics.csv", "text/csv")
