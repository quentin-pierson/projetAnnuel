from flask import Flask, render_template
import json

app = Flask(__name__)

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
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/')
def result():
    return render_template("")


if __name__ == '__main__':
    app.run(debug=True)