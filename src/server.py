import sqlite3
from mod.helper import sqlite_dict_factory
from flask import Flask, render_template, jsonify, send_file, request, Response
from PIL import Image
import cv2
import numpy as np
from mod.model import model_prob
from mod.similarity import similar_images

app = Flask(__name__, template_folder="../web/templates", static_url_path="/static", static_folder="../web/static")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # max upload size 16 megabytes

def get_connection():
    cxn = sqlite3.connect("main.db")
    cxn.row_factory = sqlite_dict_factory
    return cxn

def auth_check(auth):
    if auth is None:
        return False
    return auth.username == "ucla" and auth.password == "223c"

def auth_fail():
    return Response("access denied", 401, {"WWW-Authenticate": "Basic realm=\"Login Required\""})

@app.route("/api/query-image", methods=["POST"])
def route_api_query_image():
    f = request.files["file"]
    pil_img = Image.open(f)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    prob = model_prob(img)
    similarities = similar_images(img)
    return jsonify([img.shape[0], img.shape[1]])

@app.route("/")
def route_index():
    if not auth_check(request.authorization):
        return auth_fail()
    return render_template("index.html")
