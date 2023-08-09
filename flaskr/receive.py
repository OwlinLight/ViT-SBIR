import functools
import os

from flask import Blueprint, app
from flask import flash
from flask import g
from flask import redirect
from flask import request
from base64 import decodestring
from PIL import Image
from result import get_result

bp = Blueprint("receive", __name__)


@app.route("/uploadImage/no", methods=("POST", "OPTIONS"))
def upload():
    """Show all the posts, most recent first."""
    if request.method == 'POST':
        # check if the post request has the file part
        imagestr = request.json['image']
        width, height = 256, 256
        image = Image.fromstring('RGB', (width, height), decodestring(imagestr))
        image.save("foo.png")        # if user does not select file, browser also
        # submit a empty part without filename
        if image.filename == '':
            flash('No selected file')
            return redirect(request.url)
    return 'success'

