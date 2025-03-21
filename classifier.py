import os
import base64
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

# Point to your local LM Studio endpoint
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def classify_image(image_path):
    image_base64 = encode_image(image_path)
    completion = client.chat.completions.create(
        model="gemma-3-12b-it",
        messages=[
            {"role": "system", "content": "Classify the image as 'clean' or 'messy'. Reply only with 'clean' or 'messy'."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content.strip().lower()

# Paths to images
base_dir = './images'
categories = ['clean', 'messy']
y_true, y_pred = [], []

# Classify images
for category in categories:
    folder = os.path.join(base_dir, category)
    for img_name in tqdm(os.listdir(folder), desc=f"Processing {category} images"):
        img_path = os.path.join(folder, img_name)
        prediction = classify_image(img_path)
        y_true.append(category)
        y_pred.append(prediction)
        print(f"Image: {img_name} | True: {category} | Predicted: {prediction}")

# Evaluate
precision_clean = precision_score(y_true, y_pred, pos_label='clean')
recall_clean = recall_score(y_true, y_pred, pos_label='clean')
precision_messy = precision_score(y_true, y_pred, pos_label='messy')
recall_messy = recall_score(y_true, y_pred, pos_label='messy')

print(f"\nPrecision (clean): {precision_clean:.2f}")
print(f"Recall (clean): {recall_clean:.2f}")
print(f"Precision (messy): {precision_messy:.2f}")
print(f"Recall (messy): {recall_messy:.2f}")
