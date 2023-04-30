import base64
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pickle
from tensorflow.keras.applications.vgg19 import preprocess_input
import os
import time
from tensorflow.keras.models import load_model
import re

# load the model from the saved directory
model_dir = 'models\decoder'
decoder = tf.keras.models.load_model(model_dir)

model_dir = 'models\encoder'
encoder = tf.keras.models.load_model(model_dir)

with open(r'IPYNBs\img_name_val.pkl', 'rb') as f:
    img_name_val = pickle.load(f)

with open(r'IPYNBs\cap_val.pkl', 'rb') as f:
    cap_val = pickle.load(f)
# Load the tokenizer from the file
with open(r'IPYNBs\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load max_length from the saved file
with open(r'IPYNBs\max_length.pkl', 'rb') as f:
    max_length = pickle.load(f)
image_features_extract_model = load_model('IPYNBs\image_features_extract_model.h5')
attention_features_shape = 49

import re

def replace_repeated_phrase(sentence):
    # Find all repeated phrases in the sentence
    pattern = re.compile(r'(\w+\W*[\w+\W*]*)(?:\W+\1\b)+')
    repeated_phrases = pattern.findall(sentence)

    # Replace each repeated phrase with just one occurrence
    for phrase in repeated_phrases:
        sentence = sentence.replace(f"{phrase} ", "", sentence.count(phrase)-1)

    return sentence

# define a function to plot the attention maps for each word generated

# def plot_attention(image, result, attention_plot):
#     temp_image = np.array(Image.open(image))
#     fig = plt.figure(figsize=(10, 10))
#     len_result = len(result)
#     for l in range(len_result):
#         temp_att = np.resize(attention_plot[l], (8, 8))
#         ax = fig.add_subplot(len_result//2, len_result//2, l+1)
#         ax.set_title(result[l])
#         img = ax.imshow(temp_image)
#         ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

#     plt.tight_layout()
#     plt.show()


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = tf.zeros((1, 512))
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(
            dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)
    attention_plot = attention_plot[:len(result), :]

    return result, attention_plot


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    return img, image_path




# remove repeated phrases
def replace_repeated_phrase(sentence):
    # Find all repeated phrases in the sentence
    pattern = re.compile(r'(\w+\W*[\w+\W*]*)(?:\W+\1\b)+')
    repeated_phrases = pattern.findall(sentence)

    # Replace each repeated phrase with just one occurrence
    for phrase in repeated_phrases:
        sentence = sentence.replace(f"{phrase} ", "", sentence.count(phrase)-1)

    return sentence

app = Flask(__name__)


@app.route('/upload_image/', methods=['POST'])
def upload_image():
    # Retrieve image from form data
    image = request.files['image']

    # load the image file and preprocess it
    # img = request.files['file']
    pt = r'uploads\uploaded_image.jpg'
    image.save(pt)
    image_path = pt

    start = time.time()
    # real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    result, attention_plot = evaluate(pt)
    

    # remove <start> and <end> from the real_caption
    # first = real_caption.split(' ', 1)[1]
    # real_caption = first.rsplit(' ', 1)[0]

    # remove "<unk>" in result
    for i in result:
        if i == "<unk>":
            result.remove(i)

    # for i in real_caption:
    #     if i=="<unk>":
    #         real_caption.remove(i)

    # remove <end> from result
    result_join = ' '.join(result)
    result_final = result_join.rsplit(' ', 1)[0]
    for j in range(len(result_final.split())//2):
        result_final = replace_repeated_phrase(result_final)
    # real_appn = []
    # real_appn.append(real_caption.split())
    # reference = real_appn
    # candidate = result
    # smoothie = SmoothingFunction().method4  # choose a smoothing function

    # # compute BLEU score
    # score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    # #score = sentence_bleu(reference, candidate)
    # print(f"BELU score: {score*100}")

    # print ('Real Caption:', real_caption)
    # print('Prediction Caption:', result_final)

    # Generate captions (example data)
    captions = {
        'malayalam': '',
        'english': result_final,
        'hashtags': ""
    }

    # Return captions as JSON response
    return jsonify(captions)


@app.route('/')
def index():
    return '''
    <html>

<head>
	  <title>Captionator</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <!-- Import Material UI CSS -->
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" />
  <!-- Import Material UI icons -->
  <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons" />
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Roboto', sans-serif;
        }
        .header {
            background-color: #26a69a;
            color: #fff;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .header h1 {
            font-size: 36px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 3px;
        }
    </style>

	<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
		<script>
		$(document).ready(function () {
			$("#image-upload").change(function () {
				readURL(this);
			});
		});

		function readURL(input) {
			if (input.files && input.files[0]) {
				var reader = new FileReader();

				reader.onload = function (e) {
					$('#image-preview').attr('src', e.target.result);
					$('#image-caption-malayalam').hide();
					$('#image-caption-english').hide();
					$('#image-hashtags').hide();
                    $('#captionpart').show();
                    $('#progessdiv').show();
                    $('#spinner').show(); // show spinner when image is uploaded

					// Send image file to server API
					var form_data = new FormData();
					form_data.append('image', input.files[0]);
					$.ajax({
						url: '/upload_image/',
						type: 'POST',
						data: form_data,
						contentType: false,
						cache: false,
						processData: false,
						success: function (data) {
                        	$('#image-caption-malayalam').show();
							$('#image-caption-english').show();
							$('#image-hashtags').show();
                            
							$('#image-caption-malayalam').text(data.malayalam);
							$('#image-caption-english').text(data.english);
							$('#image-hashtags').text(data.hashtags);
                            $('#spinner').hide(); // hide spinner when image is processed
                            $('#progessdiv').hide();
						}
					});
				}

				reader.readAsDataURL(input.files[0]);
			}
		}
	</script>
</head>

<body>
     <div class="header">
        <h1>&lt;!Captionator&gt;</h1>
    </div>
  <div class="container">
    <div class="row">
      <div class="col s12">
        <div class="file-field input-field">
          <div class="btn">
            <span>File</span>
            <input type="file" id="image-upload" accept=".png, .jpg, .jpeg" name="image" />
          </div>
          <div class="file-path-wrapper">
            <input class="file-path validate" type="text" placeholder="Upload an image" />
          </div>
        </div>
      </div>
      <div class="col s12">
        <div class="row">
          <div class="col s12 m6">
            <div class="card">
              <div class="card-image">
                <img id="image-preview" />
              </div>
            </div>
          </div>
          <div class="col s12 m6" id="captionpart" style="display:none;">
            <div class="card">
              <div class="card-content">
                <div id="image-caption-malayalam"></div>
                <div id="image-caption-english"></div>
                <div id="image-hashtags"></div>
              </div>
              <div class="card-action" id="progessdiv">
                <div class="progress" id="spinner" style="display:none;">
                  <div class="indeterminate"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Import Material UI JS -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script>
</body>

</html>
    '''
