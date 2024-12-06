import logging as lg

from flask_session import Session

import os

import glob

from flask import Flask

# from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# app.config.from_object("config")

app.config['SESSION_PERMANENT'] = True
app.config['SESSION_TYPE'] = 'filesystem'

Session(app)

# db = SQLAlchemy(app)

# db.init_app(app)

# @app.cli.command()
# def init_upload():
#     for directory in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static/img/upload/*")):
#         try:
#             for file_path in glob.glob(os.path.join(directory, "*")):
#                 if os.path.basename(file_path) != '0.jpg':
#                     os.remove(file_path)
#         except Exception:
#             pass
#     lg.warning("Uploading directory is clean !")

# initiate dash app
# register_dashapp(app)

# from flowers_app import models
from sla_app import views
