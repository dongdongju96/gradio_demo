import numpy
import gradio as gr
from PIL import Image
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load an official model


def image_predict_yolo(img):
    # Run inference on 'bus.jpg' with arguments
    result = model(img)

    im = result[0].plot()
    im = Image.fromarray(im)
    
    return im

demo = gr.Interface(fn= image_predict_yolo, inputs=[
            gr.Image(tool="select",label="Image to Convert", show_label=True)], outputs="image")
            
demo.launch(share=True)