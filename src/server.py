import sqlite3
from mod.helper import sqlite_dict_factory
from flask import Flask, render_template, jsonify, send_file, request, Response

app = Flask(__name__, template_folder="../web/templates", static_url_path="/static", static_folder="../web/static")
app.config["TEMPLATES_AUTO_RELOAD"] = True

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

@app.route("/")
def index():
    if not auth_check(request.authorization):
        return auth_fail()
    return render_template("index.html")
