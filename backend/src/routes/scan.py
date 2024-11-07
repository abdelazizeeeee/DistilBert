from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse
from ..utils.document_scanner.document_scanner_save_rgb import load_model, scan
import torch
import cv2
import numpy as np
import uuid
import os

router = APIRouter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "mbv3"  # or "r50" for ResNet-50
model = load_model(model_name=model_name, device=device)

# Set the directory where images will be saved (make sure this directory exists)
IMAGE_DIR = "static/processed_images"
os.makedirs(IMAGE_DIR, exist_ok=True)  # Ensure the directory exists


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

    # Process the image
    processed_image = scan(
        image=image,
        trained_model=model,
        image_size=384,
        BUFFER=10,
        threshold=11,
        c=9
    )

    # Generate a unique filename and save the processed image
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    cv2.imwrite(filepath, processed_image)

    # Construct the URL to the saved image
    image_url = f"http://localhost:8000/{IMAGE_DIR}/{filename}"

    # Return the URL as JSON
    return JSONResponse(content={"image_url": image_url})
