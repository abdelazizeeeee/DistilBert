import os
import gc
from typing import TypeAlias
from typing import cast
import cv2
import numpy as np
import torch
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import (
    DeepLabV3,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
)

Mat: TypeAlias = np.ndarray[int, np.dtype[np.generic]]

def load_model(num_classes=2, model_name="mbv3", device=torch.device("cpu")):
    """Load the DeepLabV3 model"""
    if model_name == "mbv3":
        model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True)
        checkpoint_path = os.path.join(os.getcwd(), "src/utils/document_scanner/model_mbv3_iou_mix_2C_aux.pth")
    else:
        model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
        checkpoint_path = os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")

    model.to(device)
    checkpoints = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()
    return model

def image_preprocess_transforms(
    mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)
):
    common_transforms = torchvision_T.Compose(
        [
            torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean, std),
        ]
    )
    return common_transforms

def order_points(pts):
    """Rearrange coordinates to order:
    top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype("int").tolist()

def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)

def scan(image: np.ndarray, trained_model: DeepLabV3, image_size=384, BUFFER=10, threshold=11, c=1):
    """
    Scan and process a document image.
    
    Args:
        image (np.ndarray): Input image as a NumPy array
        trained_model (DeepLabV3): Trained DeepLabV3 model
        image_size (int): Size to resize image to
        BUFFER (int): Buffer for image padding
        threshold (int): Threshold for adaptive thresholding
        c (int): C value for adaptive thresholding
    
    Returns:
        final (numpy.ndarray): Processed document image
    """
    # Initialize preprocessing transforms
    preprocess_transforms = image_preprocess_transforms()
    
    # Check if the input image is valid
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a valid NumPy array.")
    
    # Set IMAGE_SIZE and half size for padding calculations
    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2

    imH, imW, C = image.shape

    # Resize image for model processing
    image_model = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE

    # Prepare the image for the model
    image_torch = cast(torch.Tensor, preprocess_transforms(image_model))
    model_input = torch.unsqueeze(image_torch, dim=0)

    with torch.no_grad():
        out = trained_model(model_input)["out"].cpu()

    del model_input
    gc.collect()

    out = (
        torch.argmax(out, dim=1, keepdim=True)
        .permute(0, 2, 3, 1)[0]
        .numpy()
        .squeeze()
        .astype(np.int32)
    )
    r_H, r_W = out.shape

    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()

    del _out_extended
    gc.collect()

    # Edge Detection.
    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)

    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half

    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    if len(corners) > 0:  # Check if corners are detected
        print("Corners detected!")
    else:
        print("No corners detected.")

    if not (
        np.all(corners.min(axis=0) >= (0, 0))
        and np.all(corners.max(axis=0) <= (imW, imH))
    ):
        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER

        if box_x_max >= imW:
            right_pad = (box_x_max - imW) + BUFFER

        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER

        if box_y_max >= imH:
            bottom_pad = (box_y_max - imH) + BUFFER

        image_extended = np.zeros(
            (top_pad + bottom_pad + imH, left_pad + right_pad + imW, C),
            dtype=image.dtype,
        )

        image_extended[
            top_pad : top_pad + imH, left_pad : left_pad + imW, :
        ] = image
        image_extended = image_extended.astype(np.float32)

        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        image = image_extended

    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(
        np.float32(corners), np.float32(destination_corners)
    )

    final = cv2.warpPerspective(
        image,
        M,
        (destination_corners[2][0], destination_corners[2][1]),
        flags=cv2.INTER_LANCZOS4,
    )
    final = np.clip(final, a_min=0, a_max=255)
    final = final.astype(np.uint8)
    
    return final


def save_image(image: np.ndarray, output_path: str):
    """Save the processed image to the specified path"""
    cv2.imwrite(output_path, image)
    print(f"Saved processed image to: {output_path}")

# def main():
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Load model
#     model_name = "mbv3"  # or "r50" for ResNet-50
#     model = load_model(model_name=model_name, device=device)
    
#     # Process image
#     input_path = "document-scanner/test_images/IMG_20220721_162811.jpg"  
#     output_path = "document-scanner/res/res.jpg"  
    
#     # Scan document with custom parameters
#     processed_image = scan(
#         image_path=input_path,
#         trained_model=model,
#         image_size=384,
#         BUFFER=10,
#         threshold=11,  # Adjust this value as needed (1-301, odd numbers only)
#         c=9  # Adjust this value as needed (1-30)
#     )
    
#     # Save the processed image
#     save_image(processed_image, output_path)

# main()