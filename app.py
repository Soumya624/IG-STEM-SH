from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
from gtts import gTTS
from IPython.display import Audio, display
import joblib
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['AUDIO_FOLDER'] = 'static/audio/'

config_path = "VoiceVision.pkl"
config = joblib.load(config_path)
net, layer_names, output_layers, classes = None, None, None, None

def loadYolo(config):
    global net, layer_names, output_layers, classes
    net = cv2.dnn.readNet(config["model_weights"], config["model_cfg"])
    with open(config["classes_file"], "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def showImage(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def findObject(image_path):
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    description = ""
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            description += f"{label} with {confidence*100:.1f}% confidence. "

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    cv2.imwrite(result_image_path, frame)

    audio_path = None
    if description:
        tts = gTTS(description)
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], 'getObject.mp3')
        tts.save(audio_path)

    return result_image_path, description, audio_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            result_image_path, description, audio_path = findObject(filename)
            return render_template('index.html', uploaded_image=filename, result_image=result_image_path, description=description, audio_path=audio_path)
    return render_template('index.html')

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

if __name__ == "__main__":
    config = joblib.load(config_path)
    loadYolo(config)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
    app.run(debug=True)
