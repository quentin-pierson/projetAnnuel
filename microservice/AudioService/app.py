import os

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, abort, jsonify
import json
from werkzeug.utils import secure_filename
import requests
from scrapping import *
from importlib import import_module
import sys
# sys.path.insert(0, '../../python')
sys.path.insert(1, "C:/Users/Antoine/Desktop/Projet_annuel/projetAnnuel/python/")

# linearModel = import_module("python.linearModel")
# MLPModel = import_module("python.mlp")

from linearModel import linearModel
from mlp import MLPModel
#import mlp

lm = linearModel()
mlp = MLPModel()

app = Flask(__name__)

@app.route('/useExistingModel', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        convertYtb(request.form['url'])

        convertSpect("test", 1)
        jsonModel = request.form['model'].split('_')
        modelName = jsonModel[0]
        modeltype = jsonModel[1]
        if modelName == "mlp":
            resultat3 = mlp.load_mlp_model(os.path.join("C:/Users/Antoine/Desktop/Projet_annuel/projetAnnuel/Save/", request.form['model']))
            model3 = resultat3[0]
            resultImage = import_images_and_resize("music_1_test.png")
            print("resultImage : ", resultImage)
            if modeltype == "classification":
                predicted_outputs = mlp.predict_mlp_model_classification(model3, resultImage)
                print("***", predicted_outputs)
                return transformData(predicted_outputs, resultat3)

            if modeltype == "regression":
                predicted_outputs = mlp.predict_mlp_model_regression(model3, resultImage)
                print("***", predicted_outputs)
                mlp.free_MLP(resultat3)
                return transformData(predicted_outputs)

        elif modelName == "linear":
            modeltype = jsonModel[2]
            return "ok"

        elif modelName == "rbf":
            return "ok"
                #return jsonify(predicted_outputs)
            #else:
                #return "ok"

        ######## Parser cette string pour savoir quel modèle lancer
        #print("*** request: ",request.form['model'])
        #print("###############")

    else:
        return "ok"

def transformData(data, resultat3):
    test = np.argmax(data)
    result = ""
    if test == 0:
        result = "electro"
    elif test == 1:
        result = "metal"
    else:
        result = "rap"

    deletePicture()

    mlp.free_MLP(resultat3)

    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5002, debug=True)