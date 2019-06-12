from flask import Flask, render_template, jsonify, send_file, request, Response, send_from_directory
from PIL import Image
import cv2
import numpy as np
from mod.model import Classifier, Segmenter
from mod.miniVGG_FFT_hash import ImageSimilarity
from mod import preprocess
from mod import util
from mod.labels import Labels
from argparse import ArgumentParser
import tensorflow as tf
import base64
from io import BytesIO
import json

MINIVGG_MODEL_PATH = "../data/miniVGG.h5"
HASH_IMG_DIR = "../data/segs2/patches"
CLASSIFY_MODEL_1 = ("UNET architecture", "../data/lesion_classification.model")
CLASSIFY_MODEL_2 = ("VGG16 architecture", "../data/model")
SEGMENTER_MODEL_PATH = "../data/lung_seg.model"
LABEL_FILES = ["../data/Test.csv", "../data/Train.csv"]
CLASSIFY_TEST_RESULTS_FILE = ["../data/Test-result.csv"]

app = Flask(__name__, template_folder="../web/templates", static_url_path="/static", static_folder="../web/static")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # max upload size 16 megabytes
graph = tf.get_default_graph()

def main():
    global g_data
    g_data = {}

    g_data["classifier1"] = Classifier(CLASSIFY_MODEL_1[1], graph)
    g_data["classifier2"] = Classifier(CLASSIFY_MODEL_2[1], graph)
    g_data["segmenter"] = Segmenter(SEGMENTER_MODEL_PATH, graph)
    g_data["hash_similarity"] = ImageSimilarity(HASH_IMG_DIR, preprocess.preprocess, MINIVGG_MODEL_PATH, graph)
    g_data["labels"] = Labels(LABEL_FILES)

    app.run(host="0.0.0.0", port=8085)

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
    filtered_img = g_data["segmenter"].segmenter(filtered_img)

    translated_patch_coordinates = preprocess.translate_patch_coordinates(filtered_img, request_data["point"])
    patch = util.extract_patch(filtered_img, (translated_patch_coordinates["y"], translated_patch_coordinates["x"]), preprocess.PATCH_SIZE)

    filtered_img = preprocess.preprocess(filtered_img)
    prob1 = g_data["classifier1"].classify1(filtered_img)
    prob2 = g_data["classifier2"].classify2(patch)

    similarities = g_data["hash_similarity"].query_image(patch)
    similarities = g_data["labels"].add_labels_to_similarity_list(similarities)
    img_b64 = img_to_base64(filtered_img)
    patch_b64 = img_to_base64(patch)
    res = {
        "filtered_img": img_b64.decode("utf-8"),
        "patch": patch_b64.decode("utf-8"),
        "probability1": prob1,
        "probability2": prob2,
        "similar_images": similarities,
    }
    return jsonify(res)

@app.route("/similar-images/<path:filename>")
def route_similar_images(filename):
    if not auth_check(request.authorization):
        return auth_fail()
    return send_from_directory(HASH_IMG_DIR, filename)

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
