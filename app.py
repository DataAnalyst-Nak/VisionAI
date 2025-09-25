import io
import os
import keras
from PIL import Image
import tensorflow as tf
import json, numpy as np
from flask import Flask, render_template, jsonify, request


app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def index():
   if request.method == "POST":
      response = {"success": False}

      if 'file' in request.files:
         # 1. read & pre-proces image that we pick
         image = request.files["file"].read()
         image = Image.open(io.BytesIO(image))
         
         if image.mode != "RGB":
            image = image.convert("RGB")

         image = image.resize((32,32)) 
         image = tf.keras.utils.img_to_array(image) # shape (32,32,3)
         image = np.expand_dims(image, axis=0) # shape (1,32,32,3)

         # 2. load the trained model & predict on the above image
         model = keras.saving.load_model("./model/VisionAI_Model.h5")
         y_pred = model.predict(image)
         '''
         https://keras.io/api/datasets/cifar10/

         label_code  class_label
         ------------------------
         0	         airplane
         1	         automobile
         2	         bird
         3	         cat
         4	         deer
         5	         dog
         6	         frog
         7	         horse
         8	         ship
         9	         truck
         '''
         # filter the label code that has the highest probability
         label_code = np.argmax(y_pred)
         class_label = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

         response['label'] = class_label[label_code]
         response['class_proba'] = json.loads(json.dumps(y_pred[0].tolist()))
         response['class_label'] = class_label
         response['success'] = True

         return jsonify(response)
   
   return render_template('index.html')


# main function
# if __name__ == '__main__':
#    app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use PORT provided by Render
    app.run(host="0.0.0.0", port=port, debug=True)