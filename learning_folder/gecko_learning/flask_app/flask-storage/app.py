from flask import Flask, request
from flask_uploads import UploadSet, configure_uploads, IMAGES

app = Flask(__name__)
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'pictures'

configure_uploads(app, photos)


@app.route('/upload', methods=["GET","POST"])
def upload():
    if request.method == "POST":
        image_file_name = photos.save(request.files['image'])
        return f"Image {image_file_name} is stored locally"
    else:
        return "Upload is not allowed"




if __name__ == "__main__":
    app.run(debug=True)