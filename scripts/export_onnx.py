"""
LU2Net ONNX Export Script
Exports LightUNet_170.pth → ../public/models/lu2net.onnx

Usage:
  source .venv/bin/activate
  python export_onnx.py
"""

import os
import sys
import torch
import onnx
from onnx import checker

sys.path.insert(0, os.path.dirname(__file__))
from LU2Net import LU2Net

WEIGHTS = os.path.join(os.path.dirname(__file__), "LightUNet_170.pth")
OUTPUT  = os.path.join(os.path.dirname(__file__), "../public/models/lu2net.onnx")

def export():
    print("Loading model...")
    model = LU2Net()
    state = torch.load(WEIGHTS, map_location="cpu")
    # Handle both raw state_dict and checkpoint wrappers
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB fp32)")

    # Use a 256x256 dummy — dynamic axes let ORT accept any size at runtime
    dummy = torch.zeros(1, 3, 256, 256)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy,
        OUTPUT,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )

    print("Validating ONNX graph...")
    onnx_model = onnx.load(OUTPUT)
    checker.check_model(onnx_model)

    size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"Exported: {OUTPUT}")
    print(f"Size: {size_kb:.1f} KB")

    # Quick numerical validation: compare PyTorch vs ONNX outputs
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(OUTPUT, providers=["CPUExecutionProvider"])
        test_input = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            pt_out = model(test_input).numpy()

        ort_out = sess.run(["output"], {"input": test_input.numpy()})[0]

        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        print(f"Max PyTorch/ONNX diff: {max_diff:.2e}  ", end="")
        if max_diff < 1e-4:
            print("PASS")
        else:
            print("WARN — diff higher than expected, check results")
    except ImportError:
        print("onnxruntime not installed, skipping numerical validation")

    print("Done.")

if __name__ == "__main__":
    export()
