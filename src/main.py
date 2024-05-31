import os
import gradio as gr
from utils import process_image

# Example images
data_path = "data/"
example_images = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Gradio interface
inputs = gr.Image(label="Upload an Image or Select an Example")
outputs = gr.Image(label="Image with Detections")

gr.Interface(fn=process_image, 
             inputs=inputs, 
             outputs=outputs, 
             allow_flagging="never",
             examples=example_images,
             title="YOLOv8 Object Detection",
             description="Upload an image or select an example image to run YOLOv8 object detection and visualize the results.",
             theme="default").launch(
                 share=True,
                 server_name="127.0.0.1", 
                 server_port=8000)
