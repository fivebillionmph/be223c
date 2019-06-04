from flask import Flask, render_template, jsonify, send_file, request, Response, send_from_directory
from PIL import Image
import cv2
import numpy as np
from mod.model import Model
from mod.similarity import similar_images
from mod import preprocess
from mod import util
from argparse import ArgumentParser
import tensorflow as tf
import base64
from io import BytesIO
import json

app = Flask(__name__, template_folder="../web/templates", static_url_path="/static", static_folder="../web/static")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # max upload size 16 megabytes
graph = tf.get_default_graph()

def main():
    global g_model

    args = get_args()
    g_model = Model(args.m, graph)
    app.run(host="0.0.0.0", port=8085)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-m")
    args = parser.parse_args()
    return args

def auth_check(auth):
    if auth is None:
        return False
    return auth.username == "ucla" and auth.password == "223c"

def auth_fail():
    return Response("access denied", 401, {"WWW-Authenticate": "Basic realm=\"Login Required\""})

def img_to_base64(img):
    img = Image.fromarray(np.uint8(img*255))
    b = BytesIO()
    img.save(b, format="PNG")
    b64 = base64.b64encode(b.getvalue())
    return b64

@app.route("/api/query-image", methods=["POST"])
def route_api_query_image():
    if not auth_check(request.authorization):
        return auth_fail()
    request_data = json.loads(request.form["data"])
    f = request.files["file"]
    pil_img = Image.open(f)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    filtered_img = preprocess.preprocess(img)
    patch = util.extract_patch(filtered_img, (request_data["point"]["y"], request_data["point"]["x"]), preprocess.PATCH_SIZE)

    prob = g_model.prob(filtered_img)
    similarities = similar_images(filtered_img)
    img_b64 = img_to_base64(filtered_img)
    patch_b64 = img_to_base64(patch)
    res = {
        "filtered_img": img_b64.decode("utf-8"),
        "patch": patch_b64.decode("utf-8"),
        "probability": prob,
        "similar_images": similarities,
    }
    return jsonify(res)

@app.route("/similar-images/<path:filename>")
def route_similar_images(filename):
    if not auth_check(request.authorization):
        return auth_fail()
    return send_from_directory("../data/segs/patches", filename)

@app.route("/")
def route_index():
    if not auth_check(request.authorization):
        return auth_fail()
    page_data = {
        "width": preprocess.WIDTH,
        "height": preprocess.HEIGHT,
    }
    return render_template("index.html", data=page_data)

main()
