from os import listdir
from os.path import isfile, join
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def savedModel():
    mypath = "C:/Users/Antoine/Desktop/Projet_annuel/projetAnnuel/Save"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return render_template('savedModel.html', file=onlyfiles)

@app.route('/launchModel')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return "ok c'est un test"

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001, debug=True)