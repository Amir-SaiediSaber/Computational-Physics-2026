"""Train the paired YOLO models and cross-evaluate the split protocols.

For each class (impact_crater, wrinkle_ridge):
  1. train yolov8n on the random-split dataset   -> eval on its own val
  2. train yolov8n on the spatial-split dataset  -> eval on its own val
  3. cross-eval: the random-split model scored on the SPATIAL val set
     (its generalisation to unseen terrain).

Training config mirrors the committed YOLO runs (yolov8n.pt pretrained,
imgsz 256, batch 16, seed 0); epochs reduced to 40 — identical for both
arms, so the protocol comparison is budget-matched.
"""
import os
import json
import pandas as pd
import torch
from ultralytics import YOLO

HERE = os.path.abspath(os.path.dirname(__file__))
EPOCHS = 40
CLASSES = ["impact_crater", "wrinkle_ridge"]
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

results = []


def val_metrics(model, data_yaml, tag):
    m = model.val(data=data_yaml, imgsz=256, device=DEVICE, verbose=False)
    row = {
        "run": tag,
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }
    print(json.dumps(row), flush=True)
    results.append(row)


for cname in CLASSES:
    trained = {}
    for arm in ["random", "spatial"]:
        data_yaml = os.path.join(HERE, f"{cname}_{arm}", "data.yaml")
        run_name = f"{cname}_{arm}"
        best_path = os.path.join(HERE, "runs", run_name, "weights", "best.pt")
        last_path = os.path.join(HERE, "runs", run_name, "weights", "last.pt")
        done_marker = os.path.join(HERE, "runs", run_name, "DONE")
        print(f"\n=== TRAIN {run_name} ===", flush=True)
        if os.path.exists(done_marker):
            print("already completed, skipping training", flush=True)
        elif os.path.exists(last_path):
            print(f"resuming from {last_path}", flush=True)
            YOLO(last_path).train(resume=True)
            open(done_marker, "w").close()
        else:
            model = YOLO("yolov8n.pt")
            model.train(
                data=data_yaml,
                epochs=EPOCHS,
                imgsz=256,
                batch=16,
                seed=0,
                device=DEVICE,
                project=os.path.join(HERE, "runs"),
                name=run_name,
                exist_ok=True,
                verbose=False,
                plots=True,
            )
            open(done_marker, "w").close()
        best = YOLO(best_path)
        val_metrics(best, data_yaml, f"{run_name}__own_val")
        trained[arm] = best

    # Cross-protocol: random-trained model on the spatial (unseen-terrain) val
    spatial_yaml = os.path.join(HERE, f"{cname}_spatial", "data.yaml")
    val_metrics(trained["random"], spatial_yaml,
                f"{cname}_random__spatial_val")

    pd.DataFrame(results).to_csv(os.path.join(HERE, "pair_results.csv"),
                                 index=False)

print("\nALL DONE")
print(pd.DataFrame(results).to_string(index=False))
