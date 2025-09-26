import base64
import io
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image


app = FastAPI(title="LaMa Inpainting API", version="1.0.0")


class InpaintRequest(BaseModel):
    image: str  # base64-encoded image
    mask: str   # base64-encoded binary mask (255 where to inpaint)


def _decode_b64_image(data_b64: str) -> Image.Image:
    try:
        # Allow data URLs like "data:image/png;base64,...."
        if data_b64.startswith("data:"):
            data_b64 = data_b64.split(",", 1)[1]
        raw = base64.b64decode(data_b64, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


def _save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert RGBA to RGB for LaMa compatibility
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, format="PNG")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/inpaint")
def inpaint(req: InpaintRequest) -> dict:
    request_id = uuid.uuid4().hex[:12]
    workspace = Path("/workspace")
    lama_dir = workspace / "lama"
    model_dir = lama_dir / "big-lama"
    input_root = workspace / "input" / request_id
    output_root = workspace / "output" / request_id

    # Prepare IO dirs
    for p in [input_root, output_root]:
        p.mkdir(parents=True, exist_ok=True)

    # Decode inputs
    image = _decode_b64_image(req.image)
    mask = _decode_b64_image(req.mask)

    # Normalize masks to single channel binary 0/255
    if mask.mode != "L":
        mask = mask.convert("L")
    # Threshold any nonzero to 255
    mask = mask.point(lambda x: 255 if x > 0 else 0)

    image_path = input_root / "image.png"
    # LaMa expects mask files matching pattern *mask*.png in the same tree as images
    mask_path = input_root / "image_mask.png"

    _save_image(image, image_path)
    mask.save(mask_path)

    # Run LaMa predict
    if not model_dir.exists():
        raise HTTPException(status_code=500, detail="Model weights not found. They should be downloaded at startup.")

    cmd = [
        "python3",
        "bin/predict.py",
        f"model.path={model_dir}",
        f"indir={input_root}",
        f"outdir={output_root}",
    ]

    try:
        subprocess.run(
            cmd,
            cwd=str(lama_dir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"LaMa inference failed: {e.stderr.decode(errors='ignore')}")

    # Find first image in output
    if not output_root.exists():
        raise HTTPException(status_code=500, detail="No output produced.")

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    result_path: Optional[Path] = None
    for child in sorted(output_root.glob("**/*")):
        if child.is_file() and child.suffix.lower() in exts:
            result_path = child
            break

    if result_path is None:
        raise HTTPException(status_code=500, detail="Result image not found in output directory.")

    with open(result_path, "rb") as f:
        b64_res = base64.b64encode(f.read()).decode()

    # Cleanup per-request input/output to save disk (best-effort)
    try:
        shutil.rmtree(input_root, ignore_errors=True)
    except Exception:
        pass
    # Keep outputs for debugging? Remove to save space
    try:
        shutil.rmtree(output_root, ignore_errors=True)
    except Exception:
        pass

    return {"result": b64_res}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)


