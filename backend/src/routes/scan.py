from fastapi import APIRouter, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse
from ..utils.document_scanner.document_scanner_save_rgb import load_model, scan
import torch
import cv2
import numpy as np
from io import BytesIO
import base64

router = APIRouter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "mbv3"  # or "r50" for ResNet-50
model = load_model(model_name=model_name, device=device)


@router.post("/upload-and-process-image")
async def upload_and_process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not an image.",
        )

    image_bytes = await file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode the uploaded image.",
        )

    processed_image = scan(
        image=image,
        trained_model=model,
        image_size=384,
        BUFFER=10,
        threshold=11,
        c=9
    )

    _, image_encoded = cv2.imencode(".jpg", processed_image)
    image_bytes = image_encoded.tobytes()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{image_base64}"

    return JSONResponse(content={"data_uri": data_uri})
