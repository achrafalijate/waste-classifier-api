from flask import Flask, request, jsonify
from fastai.vision.all import *
import io
import json
import base64
from PIL import Image

app = Flask(__name__)

# Load model and config
print("Loading model...")
learn = load_learner('waste_classifier_cloud.pkl')

with open('model_config.json', 'r') as f:
    config = json.load(f)
MY_CATEGORIES = config['category_mapping']

print(f"✓ Model ready. Classes: {list(learn.dls.vocab)}")

@app.route('/')
def health():
    return jsonify({
        'status': 'Waste Classifier API',
        'classes': list(learn.dls.vocab),
        'mapping': MY_CATEGORIES
    })

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get image
        if request.content_type == 'application/json':
            data = request.get_json()
            image_bytes = base64.b64decode(data['image'])
        else:
            image_bytes = request.data
            
        # Process
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        
        # Predict
        pred, pred_idx, probs = learn.predict(img)
        original_class = str(pred)
        my_category = MY_CATEGORIES.get(original_class, 'unknown')
        
        return jsonify({
            'your_category': my_category,
            'original_prediction': original_class,
            'confidence': round(float(probs[pred_idx]), 4),
            'all_probabilities': {
                str(c): round(float(p), 4) 
                for c, p in zip(learn.dls.vocab, probs)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)