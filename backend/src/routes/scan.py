from fastapi import APIRouter, HTTPException, status, File, UploadFile
from fastapi.responses import Response
from ..utils.document_scanner.document_scanner_save_rgb import load_model, scan
import torch
import cv2
import numpy as np
from fpdf import FPDF
import tempfile
import os
from PIL import Image
import io

router = APIRouter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "mbv3"  # or "r50" for ResNet-50
model = load_model(model_name=model_name, device=device)


class CustomPDF(FPDF):
    def __init__(self, image_width, image_height):
        width_pt = image_width * 72 / 96
        height_pt = image_height * 72 / 96
        super().__init__(unit='pt', format=(width_pt, height_pt))

    def header(self):
        pass

    def footer(self):
        pass


def create_pdf_from_image(image_array: np.ndarray) -> bytes:
    """
    Convert processed image to PDF without white background

    Args:
        image_array: numpy array of the processed image

    Returns:
        bytes: PDF file as bytes
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(image_rgb)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img, \
            tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:

        pil_image.save(tmp_img.name, format='JPEG', quality=95)

        img_width, img_height = pil_image.size

        pdf = CustomPDF(img_width, img_height)
        pdf.set_margins(0, 0, 0)
        pdf.add_page()

        pdf.image(tmp_img.name, x=0, y=0, w=pdf.w, h=pdf.h)

        pdf.output(tmp_pdf.name)

        with open(tmp_pdf.name, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()

    os.unlink(tmp_img.name)
    os.unlink(tmp_pdf.name)

    return pdf_bytes


@router.post("/upload-and-process-image")
async def upload_and_process_image(file: UploadFile = File(...)):
    """
    Process uploaded image and return as PDF

    Args:
        file: Uploaded image file

    Returns:
        Response: PDF file as response with appropriate headers
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not an image.",
        )

    try:
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode the uploaded image.",
            )

        processed_image, corners = scan(
            image=image,
            trained_model=model,
            image_size=384,
            BUFFER=10,
            threshold=11,
            c=9
        )

        if len(corners) < 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not detect document corners in the image.",
            )

        pdf_bytes = create_pdf_from_image(processed_image)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                'Content-Disposition': 'attachment; filename="processed_document.pdf"'
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )
