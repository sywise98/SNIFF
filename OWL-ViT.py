# Testing OWL-ViT
import requests
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import cv2
import numpy as np

def draw_bounding_boxes(image, results):
    image_np = np.array(image)
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if score > 0.1:  # You can adjust this threshold
            box = [round(i) for i in box.tolist()]
            # Object 1 (red)
            if (label ==0):
                cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=2)
                label_text = f"{texts[0][label]}"
                cv2.putText(image_np, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            # Object 2 (blue)
            if (label ==1):
                cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
                label_text = f"{texts[0][label]}"
                cv2.putText(image_np, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return Image.fromarray(image_np)

def resize_frame(frame, scale_percent=50):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


# Load the model and processor
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Load the nano model and processor
# processor = OwlViTProcessor.from_pretrained("google/owlvit-nano-patch32")  # Replace with actual nano model identifier
# model = OwlViTForObjectDetection.from_pretrained("google/owlvit-nano-patch32")  # Replace with actual nano model identifier

# # # Load an image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Open Webcamq
cap = cv2.VideoCapture(0)

# Define the objects you want to detect
texts = [["glasses","phone","scissors"]]

frame_count = 0
process_every_n_frames = 5

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    #cv2.imshow('Camera Feed', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1
    if frame_count % process_every_n_frames == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        # Prepare inputs
        inputs = processor(text=texts, images=image, return_tensors="pt")

    #     # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

    #     # Post-process the outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]
        # print(texts[0][labe]

    #     image_with_boxes = draw_bounding_boxes(image, results)
    #     image_with_boxes.save("output_image_with_boxes.jpg")

    #     # Print results
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {texts[0][label]} at location {box}")
        
    #     frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    #     cv2.imshow('OWL-ViT Object Detection', frame)

    # else:
    #     # Display the unprocessed frame
    #     cv2.imshow('OWL-ViT Object Detection', frame)
