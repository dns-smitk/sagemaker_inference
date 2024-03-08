from flask import Flask, request
import flask
import os, cv2
import json
import boto3, time
import numpy as np

#Load in model

s3 = boto3.client('s3', region_name='us-east-1')
bucket_name = 'kinesis-stream-to-frame'
model_weight_key = 'weights/yolov7.weights'
model_config_key = 'cfg/yolov7.cfg'

# Local directory and file paths
local_path = 'src/'
local_weights_path = os.path.join(local_path, 'yolov7.weights')
local_config_path = os.path.join(local_path, 'yolov7.cfg')

# Function to download a file from S3 if it does not exist locally
def download_from_s3_if_not_exists(s3_client, bucket, key, local_file_path):
    if not os.path.exists(local_file_path):
        print(f"Downloading {local_file_path} from S3 bucket {bucket}...")
        s3_client.download_file(bucket, key, local_file_path)
    else:
        print(f"{local_file_path} already exists. Skipping download.")

# Download the model weights and config files from S3 if they don't exist locally
download_from_s3_if_not_exists(s3, bucket_name, model_weight_key, local_weights_path)
download_from_s3_if_not_exists(s3, bucket_name, model_config_key, local_config_path)

# Load the model using OpenCV
net = cv2.dnn.readNet(local_weights_path, local_config_path)
classes = ['Gun']
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time() - 11

running = True

#If you plan to use a your own model artifacts, 
#your model artifacts should be stored in /opt/ml/model/ 


# The flask app for serving predictions
app = Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    if running : status = 200 
    else : status = 443
    return flask.Response(response= '\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    
    #Process input
    # frame = input_json['input']
    # input_json = flask.request.get_json()
    image_data = request.data

    # Convert the byte data to a NumPy array
    nparr = np.frombuffer(image_data, np.uint8)

        # Decode the image data into a format OpenCV understands
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    height, width, channels = frame.shape
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Evaluating detections
    class_ids = []
    confidences = []	
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # If detection confidance is above 98% a weapon was detected
            if confidence > 0.78:

                # Calculating coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    boxes = [list(map(int, box)) for box in boxes]  # Convert each box's coordinates to int, in case they aren't already
    confidences = [float(conf) for conf in confidences]  # Ensure confidence scores are float
    class_ids = [int(cid) for cid in class_ids]  # Ensure class IDs are int

    entities = {
        'boxes': boxes,
        'confidences': confidences,  # Changed from 'confidence' to 'confidences' to match the list of all confidence values
        'class_ids': class_ids
    }

    # Transform predictions to JSON
    result = {
        'output': entities
        }

    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)