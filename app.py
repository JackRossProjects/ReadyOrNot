from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf

import stripe
import os
import base64
import io
from PIL import Image
import keras
from tensorflow.keras import backend as k
from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

# Define a flask app
app = Flask(__name__)

import pyrebase
config = {
	"apiKey": "AIzaSyDQ_xmW0kSPGZGKxwDZ-Pw1AYMp3-9_6Ns",
    "authDomain": "thisisit-d413a.firebaseapp.com",
    "databaseURL": "https://thisisit-d413a.firebaseio.com",
    "projectId": "thisisit-d413a",
    "storageBucket": "",
    "messagingSenderId": "1065583813545",
    "appId": "1:1065583813545:web:7eb887bb08a8e596bddab8"
}
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if (request.method == 'POST'):
            email = request.form['name']
            password = request.form['password']
            try:
                auth.sign_in_with_email_and_password(email, password)
                #user_id = auth.get_account_info(user['idToken'])
                #session['usr'] = user_id
                return redirect(url_for('payment'))
            except:
                unsuccessful = 'Please check your credentials'
                return render_template('index.html', umessage=unsuccessful)
    return render_template('index.html')

@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    if (request.method == 'POST'):
            email = request.form['name']
            password = request.form['password']
            auth.create_user_with_email_and_password(email, password)
            return render_template('index.html')
    return render_template('create_account.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if (request.method == 'POST'):
            email = request.form['name']
            auth.send_password_reset_email(email)
            return render_template('index.html')
    return render_template('forgot_password.html')


pub_key = 'pk_test_WPnbA5RW1oTSJOoUNmwULdgZ00524ZWHXd'
secret_key = 'sk_test_6bJmOziDLN2DzsZkJLEAtrKs00hhaStrDe'

stripe.api_key = secret_key
@app.route('/payment')
def payment():
    return render_template('payment.html', pub_key=pub_key)

@app.route('/thanks')
def thanks():
    return render_template('thanks.html')

@app.route('/predict')
def predict():
    return render_template('/home.html')

@app.route('/pay', methods=['POST'])
def pay():

    customer = stripe.Customer.create(email=request.form['stripeEmail'], source=request.form['stripeToken'])

    charge = stripe.Charge.create(
        customer=customer.id,
        amount=500,
        currency='usd',
        description='The Product'
    )

    return redirect(url_for('predict'))

export_file_url = 'https://www.dropbox.com/s/3x66dm6h52ynz8b/fullfood.pkl?raw=1'
export_file_name = 'fullfood.pkl'

classes = ['apple_pie','baby_back_ribs','baklava', 'beef_carpaccio',
           'beef_tartare', 'beet_salad', 'beignets', 'bibimbap',
           'bread_pudding', 'breakfast_burrito', 'bruschetta',
           'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
           'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry',
           'chicken_quesadilla', 'chicken_wings', 'chocolate_cake',
           'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich',
           'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
           'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
           'escargots', 'falafel', 'filet_mignon', 'fish_and_chips',
           'foie_gras', 'french_fries', 'french_onion_soup',
           'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
           'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
           'grilled_salmon', 'guacamole', 'gyoza', 'hamburger',
           'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus',
           'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich',
           'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels',
           'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai',
           'paella', 'pancakes', 'panna_cotta', 'peking_ducks',
           'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
           'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake',
           'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad',
           'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara',
           'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi',
           'tacos', 'takoyako', 'tiramisu', 'tuna_tartare', 'waffles']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})
    if prediction == 'chocolate_cake':
        print('WHORKDINDFV')


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")

# Model saved with Keras model.save()
#MODEL_PATH = 'fullfood.pkl'

# Load your trained model
#model = load_model(MODEL_PATH)

#model._make_predict_function()          # Necessary
#print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')

#def init():
#    global model
 #   model = load_model('https://www.dropbox.com/s/3x66dm6h52ynz8b/fullfood.pkl?raw=1')
#global graph
#graph = tf.get_default_graph()
    #return graph


#def model_predict(img_path, model):
#    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
 #   x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
  #  x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    #preds = model.predict(x)
    #return preds


#@app.route('/', methods=['GET'])
#def index():
    # Main page
    #return render_template('index.html')


#@app.route('/predict', methods=['GET', 'POST'])

#def upload():
#    if request.method == 'POST':
#        # Get the file from post request
#        f = request.files['image']

        # Save the file to ./uploads
#        basepath = os.path.dirname(__file__)
 #       file_path = os.path.join(
  #          basepath, 'uploads', secure_filename(f.filename))
   #     f.save(file_path)

        # Make prediction
        #graph = tf.get_default_graph()
    #    with graph.as_default():
   #             preds = model_predict(file_path, model)

            # Process your result for human
  #              pred_class = preds.argmax(axis=-1)            # Simple argmax
                #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
    #            result = str(pred_class[0])
     #           return result



    #return None




#if __name__ == '__main__':
    # app.run(port=5002, debug=True)
    #app.run(debug=True)

    # Serve the app with gevent
    #
    #http_server = WSGIServer(('0.0.0.0', 5000), app)
    #http_server.serve_forever()
