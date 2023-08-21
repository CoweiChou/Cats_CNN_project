import os
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PIL


app = Flask(__name__)


MODEL_PATH = 'model.pth'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(MODEL_PATH, map_location=device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model(img)
        return features.numpy().flatten()
    except PIL.UnidentifiedImageError:
        print(f"Skipped non-image file: {img_path}")



# 辨識花色前處理函數
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # 添加 batch 維度
    return image

# 提取數據集中所有圖像的特徵並存儲在內存中
dataset_path = 'dataset'
features = []
img_paths = []
for img_name in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img_name)
    img_features = extract_features(img_path)
    if img_features is not None:
        features.append(img_features)
        img_paths.append(img_name)
features = np.array(features)


# 辨識貓咪花色
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('/tmp', filename)
        file.save(filepath)
        image = preprocess_image(filepath)
        with torch.no_grad():
            input_batch = image.to(device)
            output = model(input_batch)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = predicted_idx.item()

        class_labels = ['黑貓', '白貓', '橘貓', '黑白貓', '三花貓', '虎斑貓', '玳瑁貓']
        predicted_label = class_labels[predicted_class]

        return jsonify(predicted_label),200


    return jsonify({'error': 'Something went wrong'}), 500



# 辨識回傳3張最相似貓咪
@app.route('/similar_images', methods=['POST'])
def similar_images():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('/tmp', filename)
        file.save(filepath)
        user_img_features = extract_features(filepath)

        similarity = cosine_similarity(user_img_features.reshape(1,-1), features)
        top3_idx = similarity.argsort()[0][::-1][:3]
        top3_img_names = [img_paths[i] for i in top3_idx]

        return jsonify(top3_img_names)

    return jsonify({'error': 'Something went wrong'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
