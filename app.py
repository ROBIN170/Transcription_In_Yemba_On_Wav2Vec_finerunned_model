from flask import Flask, request, jsonify, render_template
import torch
import librosa
import io
from scipy.signal import butter, lfilter
import noisereduce as nr
from transformers import AutoProcessor, AutoModelForCTC

app = Flask(__name__)

# Load the processor and model
processor = AutoProcessor.from_pretrained("RoinsonNgeukeu237/working")
model = AutoModelForCTC.from_pretrained("RoinsonNgeukeu237/working", ignore_mismatched_sizes=True)

# Adjust classification head if necessary
vocab_size = 28  # Replace with the correct vocabulary size
hidden_size = model.lm_head.in_features

if model.lm_head.out_features != vocab_size:
    model.lm_head = torch.nn.Linear(hidden_size, vocab_size)


# High-pass filter function
def highpass_filter(data, sr, cutoff=1000):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = butter(1, norm_cutoff, btype='high', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')



@app.route('/transcribe', methods=['POST'])
def transcribe():
    file = request.files['file']
    # if not file or not file.filename.endswith('.mp3'):
    #     return jsonify({'error': 'Invalid file format. Please upload an MP3 file.'}), 400

    # Load and preprocess the MP3 audio using librosa
    audio, sampling_rate = librosa.load(file, sr=16000)


    # Perform noise reduction
    reduced_noise_audio = nr.reduce_noise(y = audio, sr = sampling_rate)

    filtered_audio = highpass_filter(reduced_noise_audio, sampling_rate, cutoff=1000)


    input_values = processor(filtered_audio, sampling_rate=sampling_rate, return_tensors="pt").input_values

    # Move model and data to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_values = input_values.to(device)

    # Perform prediction
    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    predicted_text = processor.batch_decode(pred_ids)[0]

    return jsonify({'transcription': predicted_text})

if __name__ == '__main__':
    app.run(debug=True)



