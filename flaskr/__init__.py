import base64
import json
import os
from base64 import decodestring, b64decode
import calendar
import datetime, time

from flask import Flask, request
from PIL import Image
from flask_cors import CORS
from .result import SBIR

def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, instance_relative_config=True, static_folder="../Dataset")
    CORS(app)
    app.config.from_mapping(
        # a default secret that should be overridden by instance config
        SECRET_KEY="dev",
        # store the database in the instance folder
        DATABASE=os.path.join(app.instance_path, "flaskr.sqlite"),
    )


    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.update(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass



    @app.route("/hello")
    def hello():
        return "Hello, World!"

    @app.route("/uploadImage/no", methods=["POST"])
    def upload():
        """Show all the posts, most recent first."""
        if request.method == 'POST':
            # check if the post request has the file part
            imagestr = request.json['image']
            imagestr = imagestr.split(",")[1]
            # width, height = 256, 256
            # image = Image.frombytes('RGB', (width, height), b64decode(imagestr))
            # image.save("foo.png")  # if user does not select file, browser also
            img = base64.b64decode(imagestr)
            filename = str(calendar.timegm(time.gmtime()))
            des = 'uploads'
            filepath = os.path.join('Dataset', 'ShoeV2', des, filename +  '.png')
            fh = open(filepath, "wb")
            fh.write(img)
            fh.close()
            ranking_dict = sbir.get_result(des, filename)
            prefix = 'http://localhost:5000/Dataset/ShoeV2/photo/'
            suffix = '.png'
            path_dict = [(prefix + k + suffix , v) for k,v in ranking_dict]
            print(json.dumps(path_dict))
        return json.dumps(path_dict)

    # register the database commands
    from flaskr import db

    db.init_app(app)

    # apply the blueprints to the app
    from flaskr import auth, blog

    app.register_blueprint(auth.bp)
    app.register_blueprint(blog.bp)

    sbir = SBIR()
    sbir.initialize_model()

    print("finish initializing")


    # make url_for('index') == url_for('blog.index')
    # in another app, you might define a separate main index here with
    # app.route, while giving the blog blueprint a url_prefix, but for
    # the tutorial the blog will be the main index
    app.add_url_rule("/", endpoint="index")

    return app
