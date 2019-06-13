"""
written by James Go

this is the main script for running the Flask server
"""

from flask import Flask, render_template, jsonify, send_file, request, Response, send_from_directory, abort
from PIL import Image
import cv2
import numpy as np
from mod.model import Classifier1, Classifier2, Segmenter
from mod.miniVGG_FFT_hash import ImageSimilarity
from mod import preprocess
from mod import util
from mod.labels import Labels
from argparse import ArgumentParser
import tensorflow as tf
import base64
from io import BytesIO
import json
from os.path import join as opj

"""
these are hard coded CONSTANTS for locations of models, images and test results
"""
MINIVGG_MODEL_PATH = "../data/miniVGG.h5"
HASH_IMG_DIR = "../data/segs2/patches-training"
CLASSIFY_MODEL_1 = ("UNET architecture", "../data/lesion_classification.model")
CLASSIFY_MODEL_2 = ("VGG16 architecture", "../data/model_lung_pro_patch3.h5")
SEGMENTER_MODEL_PATH = "../data/lung_seg.model"
LABEL_FILES = ["../data/Test.csv", "../data/Train.csv"]
CLASSIFY_TEST_RESULTS_FILE = ["../data/Test-result.csv"]
MODEL1_TEST_DIR = "../data/test-model1"
MODEL2_TEST_DIR = "../data/test-model2"
SERVER_PORT = 8085

app = Flask(__name__, template_folder="../web/templates", static_url_path="/static", static_folder="../web/static")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # max upload size 16 megabytes
graph = tf.get_default_graph()

def main():
    """
    loads the models and saves them in a global dictionary variable
    then starts the server on port 8085
    """
    global g_data
    g_data = {}

    g_data["classifier1"] = Classifier1(CLASSIFY_MODEL_1[1], graph)
    g_data["classifier2"] = Classifier2(CLASSIFY_MODEL_2[1], graph)
    g_data["segmenter"] = Segmenter(SEGMENTER_MODEL_PATH, graph)
    g_data["hash_similarity"] = ImageSimilarity(HASH_IMG_DIR, preprocess.preprocess, MINIVGG_MODEL_PATH, graph)
    g_data["labels"] = Labels(LABEL_FILES)

    app.run(host="0.0.0.0", port=SERVER_PORT)

def auth_check(auth):
    """
    used for authenticating users

    Args:
        auth: an authentication header variable

    Returns:
        bool -> if authentication is successful or not
    """
    if auth is None:
        return False
    return auth.username == "ucla" and auth.password == "223c"

def auth_fail():
    """
    Returns:
        flask Response object with 401 status code, asking for long basic http authentication
    """
    return Response("access denied", 401, {"WWW-Authenticate": "Basic realm=\"Login Required\""})

def img_to_base64(img):
    """
    converts an image to base64 encoded png

    Args:
        img: numpy array of the image

    Returns:
        byte string of the base64 encoded image
    """
    img = Image.fromarray(np.uint8(img*255))
    b = BytesIO()
    img.save(b, format="PNG")
    b64 = base64.b64encode(b.getvalue())
    return b64

@app.route("/api/query-image", methods=["POST"])
def route_api_query_image():
    """
    api endpoint to query an image
    this is the 'main' api endpoint for the application

    reads two pieces of data from the request: "data" and "file"
    the file is the binary image
    the data is json payload of other request data,
        which is just the lesion point clicked on by the use

    Returns:
        json response of:
            the two probabilites from two models
            the filtered and segmented image
            the extracted patch
            list of similar images
    """
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
    prob1 = g_data["classifier1"].classify(filtered_img)
    prob2 = g_data["classifier2"].classify(patch)

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
    """
    api endpoint for getting similar images in the HASH_IMG_DIR directory

    Args:
        filename: string of the file requested

    Returns:
        the image file specified in the api path
    """
    if not auth_check(request.authorization):
        return auth_fail()
    return send_from_directory(HASH_IMG_DIR, filename)

@app.route('/modelimg/<int:modelid>/<path:filename>')
def route_modelimg(modelid, filename):
    """
    api endpoint for getting a results image for the two models

    Args:
        modelid: int of which of the two models requested
        filename: the filename requested

    Returns:
        the file specified in the model's MODEL_TEST_DIR directory
    """
    if not auth_check(request.authorization):
        return auth_fail()
    if modelid == 1:
        dire = MODEL1_TEST_DIR
    elif modelid == 2:
        dire = MODEL2_TEST_DIR
    else:
        abort(404)
        return
    return send_from_directory(dire, filename)

@app.route('/model/<int:modelid>')
def route_model(modelid):
    """
    page request to view more information about a model

    Args:
        modelid: int of which of the two models requested

    Returns:
        rendered HTML page of the model information
    """
    if not auth_check(request.authorization):
        return auth_fail()
    page_data = {}
    if modelid == 1:
        dire = MODEL1_TEST_DIR
    elif modelid == 2:
        dire = MODEL2_TEST_DIR
    else:
        abort(404)
        return
    with open(opj(dire, "description.html")) as f:
        page_data["description"] = f.read()
    with open(opj(dire, "stats.json")) as f:
        page_data["stats"] = json.load(f)
    page_data["modelid"] = modelid
    return render_template("model.html", data=page_data)

@app.route("/")
def route_index():
    """
    server index page

    Returns:
        rendered HTML of the website's main page
    """
    if not auth_check(request.authorization):
        return auth_fail()
    page_data = {
        "width": preprocess.WIDTH,
        "height": preprocess.HEIGHT,
    }
    return render_template("index.html", data=page_data)

main()
