import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageColor
import math
import matplotlib.pyplot as plt

WORK = "model"
MODEL = "ssd_mobilenet_v1_13-qdq"

img = Image.open("data/people.jpg")
img = img.resize((150, 100))
img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
# img_data = np.stack([img_data, img_data], axis=0)
print("Image shape:", img_data.shape)

import onnxruntime as rt
sess = rt.InferenceSession(os.path.join(WORK, MODEL + ".onnx"))

# we want the outputs in this order
outputs = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
start = time.time()
result = sess.run(outputs, {"inputs": img_data})
print(time.time()-start, "s")
num_detections, detection_boxes, detection_scores, detection_classes = result

print("No. of detection: ", num_detections)
print(detection_boxes)


def draw_detection(draw, d, c):
    """Draw box and label for 1 detection."""
    width, height = draw.im.size
    # the box is relative to the image size so we multiply with height and width to get pixels.
    top = d[0] * height
    left = d[1] * width
    bottom = d[2] * height
    right = d[3] * width
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
    right = min(width, np.floor(right + 0.5).astype('int32'))
    label = int(c)
    # label_size = draw.textlength(label)
    # if top - label_size[1] >= 0:
    #     text_origin = tuple(np.array([left, top - label_size[1]]))
    # else:
    #     text_origin = tuple(np.array([left, top + 1]))
    color = ImageColor.getrgb("red")
    thickness = 0
    draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color)
    # draw.text(text_origin, label, fill=color)  # , font=font)


batch_size = num_detections.shape[0]
draw = ImageDraw.Draw(img)
for batch in range(0, batch_size):
    for detection in range(0, int(num_detections[batch])):
        c = detection_classes[batch][detection]
        d = detection_boxes[batch][detection]
        draw_detection(draw, d, c)

plt.figure(figsize=(80, 40))
plt.axis('off')
plt.imshow(img)
plt.show()