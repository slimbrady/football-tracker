# Football 22-Player Tracker

Red = Offense, Blue = Defense. Click-to-include before play starts.

### Stack
- YOLOv8 (Ultralytics) for person detection
- OC-SORT / BYTETracker for lock-on tracking IDs
- OpenCV for UI
- Field mask to ignore sidelines

### Quick start
```bash
cd football-tracker
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tracker.py --video /path/to/play.mp4
```

Use `tracker_v2.py` for the full feature set: jersey OCR, speed trails, ball tracking, position filters (DL/LB/DB/OL/WR/QB/RB), distance measurement between two players, CSV export.

```bash
python tracker_v2.py --video play.mp4 --model yolov8m.pt --ocr --px_per_yard 12.3
```

### How it works
1. First frame: detections run, 22 field players auto-proposed
2. Selection screen opens:
   - Click any box to toggle include/exclude
   - R = set to Offense (Red)
   - B = set to Defense (Blue)
   - 1-7 tag position: 1=DL 2=LB 3=DB 4=OL 5=WR 6=QB 7=RB
   - SPACE = lock in and start tracking
   - Q = quit
3. Tracking runs: boxes stay locked to their IDs via OC-SORT, no ID drift swapping
4. Sideline people are filtered out by the field polygon mask

Field polygon is auto-estimated on first frame (green HSV mask), you can also draw it manually: `python tracker_v2.py --video play.mp4 --draw_field`

Output saved to `output_tracked.mp4`, tracks to `tracks.csv`

### Tracking hotkeys
- `1-7` filter to DL/LB/DB/OL/WR/QB/RB
- `0` show all
- `G` DL+LB combo
- `H` Secondary only
- `J` WR vs CB matchup
- `D` then click 2 players to measure distance
- `C` clear measure
- `Q` quit

### Tips for clean locks
- Use a 1080p endzone/sideline angle if possible
- YOLOv8m or YOLOv8l: `--model yolov8l.pt` for better ID stability
- If IDs still swap: lower `--iou_thresh 0.3` and increase `--track_buffer 60`
- Default tracker is OC-SORT, best for sports occlusion. Swap to BYTETrack with `--tracker bytetrack.yaml`
