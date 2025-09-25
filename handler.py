import base64
import io
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Any

import runpod
from PIL import Image


def _decode_image(b64: str) -> Image.Image:
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data))


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    # event["input"] contains { image: base64, mask: base64 }
    payload = event.get("input", {})
    image_b64 = payload.get("image")
    mask_b64 = payload.get("mask")
    if not image_b64 or not mask_b64:
        return {"error": "Missing 'image' or 'mask' base64"}

    workspace = Path("/workspace")
    lama_dir = workspace / "lama"
    model_dir = lama_dir / "big-lama"
    request_id = uuid.uuid4().hex[:12]
    input_root = workspace / "input" / request_id
    output_root = workspace / "output" / request_id
    (input_root / "mask").mkdir(parents=True, exist_ok=True)
    (input_root / "masks").mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    image = _decode_image(image_b64).convert("RGB")
    mask = _decode_image(mask_b64)
    if mask.mode != "L":
        mask = mask.convert("L")
    mask = mask.point(lambda x: 255 if x > 0 else 0)

    image_path = input_root / "image.png"
    mask_path1 = input_root / "mask" / "image.png"
    mask_path2 = input_root / "masks" / "image.png"
    image.save(image_path)
    mask.save(mask_path1)
    mask.save(mask_path2)

    if not model_dir.exists():
        return {"error": "Model weights not found"}

    cmd = [
        "python3",
        "bin/predict.py",
        f"model.path={model_dir}",
        f"indir={input_root}",
        f"outdir={output_root}",
    ]
    try:
        subprocess.run(cmd, cwd=str(lama_dir), check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return {"error": f"LaMa failed: {e.stderr.decode(errors='ignore')[:2000]}"}

    result_path = None
    for p in output_root.rglob("*.png"):
        result_path = p
        break
    if not result_path:
        return {"error": "Result not found"}

    with open(result_path, "rb") as f:
        res_b64 = base64.b64encode(f.read()).decode()

    # Cleanup
    shutil.rmtree(input_root, ignore_errors=True)
    shutil.rmtree(output_root, ignore_errors=True)
    return {"result": res_b64}


runpod.serverless.start({"handler": handler})


