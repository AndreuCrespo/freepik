from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Cargar el modelo y el tokenizer
model_name = "microsoft/git-base-textcaps"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate title from the image
def generate_title(image_content):
    inputs = tokenizer("Imagen: " + image_content, return_tensors="pt", max_length=40, padding=True, truncation=True)
    with torch.no_grad():
        output = model.generate(**inputs)
    title = tokenizer.decode(output[0], skip_special_tokens=True)
    return title

@app.route('/generate_title', methods=['POST'])
def generate_image_title():
    data = request.get_json()
    image_content = data.get('image_content', '')

    if not image_content:
        return jsonify({'error': 'Se requiere el contenido de la imagen'})

    title = generate_title(image_content)
    return jsonify({'title': title})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
