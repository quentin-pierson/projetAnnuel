import os
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename

########################################################################
##############################   TASK   ################################
########################################################################

# TODO : Faire un page d'upload de vidéo 
# entrée : 
#   - V1: lien url
#   - V2: fichier audio (mp3,wav, etc...)
# sorti :  résultat de l'algo

########################################################################
##############################   CODE   ################################
########################################################################



UPLOAD_FOLDER = 'C:/Users/Antoine/Desktop/Projet_annuel/projetAnnuel/flaskApp'
ALLOWED_EXTENSIONS = ['.png', '.pdf']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['UPLOAD_PATH'] = 'upload'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    return "metal"

    # uploaded_file = request.files['file']
    # filename = secure_filename(uploaded_file.filename)
    # if filename != '':
    #     file_ext = os.path.splitext(filename)[1]
    #     if file_ext not in app.config['UPLOAD_EXTENSIONS']:
    #         abort(400)
    #     uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    # return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)