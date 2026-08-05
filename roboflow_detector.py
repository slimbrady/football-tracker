# roboflow_detector.py
# Roboflow Workflow detector for football-tracker
# Workspace: brady-powell
# Workflow: football-player-and-ball-tracker-1785894897417

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "brady-powell")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "football-player-and-ball-tracker-1785894897417")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

_inference_client = None

def get_client():
    global _inference_client
    if _inference_client is not None:
        return _inference_client
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as e:
        raise RuntimeError("pip install inference") from e
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY not set – put it in .env")
    _inference_client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY
    )
    return _inference_client

def run_workflow_frame(frame_bgr: np.ndarray) -> Dict[str, Any]:
    """Run brady-powell football-player-and-ball-tracker workflow on one BGR frame.
    Returns raw workflow result.
    """
    client = get_client()
    # inference_sdk accepts numpy array directly
    result = client.run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_WORKFLOW_ID,
        images={"image": frame_bgr}
    )
    # run_workflow returns a list when images is a dict, unwrap
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    return result

def parse_workflow_detections(workflow_result: Dict[str, Any]) -> Tuple[List[Tuple[float,float,float,float,float,int,str]], List[Tuple[float,float,float,float,float]]]:
    """
    Convert Workflow output to tracker_v2-compatible detections.
    Returns:
      players: list of (x1,y1,x2,y2, conf, class_id, class_name)
      balls:   list of (x1,y1,x2,y2, conf)
    The exact output keys depend on your Workflow blocks – this parser tries common shapes:
      - result["predictions"] / result["detections"]
      - result["player_predictions"] / result["ball_predictions"]
      - result["output_image"] etc (ignored)
    Adjust the key names below to match your Workflow if needed.
    """
    players = []
    balls = []

    def _extract_boxes(obj, default_class="player"):
        out = []
        if obj is None:
            return out
        # Case 1: {"predictions": [{x,y,width,height,confidence,class}, ...]}
        preds = None
        if isinstance(obj, dict):
            preds = obj.get("predictions") or obj.get("detections") or obj.get("boxes")
        elif isinstance(obj, list):
            preds = obj
        if not preds:
            return out
        for p in preds:
            # Roboflow JSON is usually x_center, y_center, width, height
            if all(k in p for k in ("x","y","width","height")):
                cx, cy = float(p["x"]), float(p["y"])
                bw, bh = float(p["width"]), float(p["height"])
                x1 = cx - bw/2; y1 = cy - bh/2
                x2 = cx + bw/2; y2 = cy + bh/2
            elif all(k in p for k in ("x_min","y_min","x_max","y_max")):
                x1, y1, x2, y2 = float(p["x_min"]), float(p["y_min"]), float(p["x_max"]), float(p["y_max"])
            else:
                continue
            conf = float(p.get("confidence", p.get("conf", 0.9)))
            cls_name = str(p.get("class", p.get("class_name", default_class))).lower()
            cls_id = int(p.get("class_id", 0))
            out.append((x1, y1, x2, y2, conf, cls_id, cls_name))
        return out

    # Try common top-level keys – edit these to match your Workflow
    search_keys_players = ["player_predictions", "players", "predictions", "detections", "output"]
    search_keys_ball = ["ball_predictions", "ball", "football"]

    for k in search_keys_players:
        if k in workflow_result:
            players.extend(_extract_boxes(workflow_result[k], "player"))
            if players: break

    # if still empty, try parsing the whole result as a prediction blob
    if not players:
        players = _extract_boxes(workflow_result, "player")

    for k in search_keys_ball:
        if k in workflow_result:
            b = _extract_boxes(workflow_result[k], "ball")
            balls = [(x1,y1,x2,y2,conf) for x1,y1,x2,y2,conf,_,_ in b]
            break

    # split out anything labelled "ball"/"football" that ended up in players
    filtered_players = []
    for x1,y1,x2,y2,conf,cid,cname in players:
        if "ball" in cname or "football" in cname:
            balls.append((x1,y1,x2,y2,conf))
        else:
            filtered_players.append((x1,y1,x2,y2,conf,cid,cname))
    
    return filtered_players, balls
