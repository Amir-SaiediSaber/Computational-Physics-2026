"""Extended-budget ridge pair: does the random-split score grow with epochs?

Trains the wrinkle-ridge detector for 150 epochs under both protocols
(otherwise identical to train_pair.py). If leakage inflates scores through
memorisation, the random-split validation curve should keep climbing with
budget while the spatial-split curve plateaus.
"""
import os
import json
import torch
from ultralytics import YOLO

HERE = os.path.abspath(os.path.dirname(__file__))
EPOCHS = 150
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

for arm in ["random", "spatial"]:
    run_name = f"wrinkle_ridge_{arm}_e{EPOCHS}"
    done_marker = os.path.join(HERE, "runs", run_name, "DONE")
    last_path = os.path.join(HERE, "runs", run_name, "weights", "last.pt")
    data_yaml = os.path.join(HERE, f"wrinkle_ridge_{arm}", "data.yaml")
    print(f"\n=== TRAIN {run_name} ===", flush=True)
    if os.path.exists(done_marker):
        print("already completed", flush=True)
    elif os.path.exists(last_path):
        YOLO(last_path).train(resume=True)
        open(done_marker, "w").close()
    else:
        YOLO("yolov8n.pt").train(
            data=data_yaml, epochs=EPOCHS, imgsz=256, batch=16, seed=0,
            device=DEVICE, project=os.path.join(HERE, "runs"),
            name=run_name, exist_ok=True, verbose=False, plots=True)
        open(done_marker, "w").close()
    best = YOLO(os.path.join(HERE, "runs", run_name, "weights", "best.pt"))
    m = best.val(data=data_yaml, imgsz=256, device=DEVICE, verbose=False)
    print(json.dumps({"run": f"{run_name}__own_val",
                      "map50": float(m.box.map50)}), flush=True)

# Cross-eval: long random model on spatial val
best_r = YOLO(os.path.join(HERE, "runs", f"wrinkle_ridge_random_e{EPOCHS}",
                           "weights", "best.pt"))
m = best_r.val(data=os.path.join(HERE, "wrinkle_ridge_spatial", "data.yaml"),
               imgsz=256, device=DEVICE, verbose=False)
print(json.dumps({"run": f"wrinkle_ridge_random_e{EPOCHS}__spatial_val",
                  "map50": float(m.box.map50)}), flush=True)
print("LONG DONE")
