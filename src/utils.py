from ultralytics import YOLO
import cv2


def load_yolov8_model(weights_path):
    """
    Load the YOLOv8 model with the given weights.

    Parameters:
    weights_path (str): Path to the YOLOv8 weights file.

    Returns:
    model: Loaded YOLOv8 model.
    """
    model = YOLO(weights_path)
    return model


def run_inference(model, image):
    """
    Run inference on an input image using the YOLOv8 model.

    Parameters:
    model: Loaded YOLOv8 model.
    image: Input image in numpy array format.

    Returns:
    results: YOLOv8 inference results.
    """
    results = model(image)
    return results


def display_results(image, results):
    """
    Display the inference results on the input image.

    Parameters:
    image: Input image in numpy array format.
    results: YOLOv8 inference results.

    Returns:
    image_with_detections: Image with detections visualized.
    """
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def process_image(image):
    """
    Process the input image by running YOLOv8 inference and returning the image with detections.

    Parameters:
    image: Input image in numpy array format.

    Returns:
    image_with_detections: Image with detections visualized.
    """
    results = run_inference(model, image)
    image_with_detections = display_results(image.copy(), results)
    return image_with_detections


# Load YOLOv8 model
weights_path = 'weights/yolov8_6.pt'
model = load_yolov8_model(weights_path)
