import os
import shutil
import logging
import requests
import itertools
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
import cv2
from scipy import misc
import numpy as np
import tensorflow as tf

from utils.face_detection import crop_face
from utils.facemodel import model


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.facenet = model.Facenet('utils/facemodel/facenet_models/20170512-110547')


DATASET_IMG_SIZE = (160, 160)
DATASET = 'img_align_celeba'


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('base.html')

    if 'user_image' not in request.files:
        return redirect(request.url)

    user_image = request.files['user_image']
    secure_path = os.path.join('uploads', secure_filename(user_image.filename))

    if not secure_path.endswith('jpg') and not secure_path.endswith('png'):
        return render_template('error.html', error='Image must be .jpg or .png')

    log.info('Saving user image to %s', secure_path)
    npimg = np.fromstring(user_image.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    log.info('Image: {}'.format(img))
    misc.imsave(secure_path, img)

    photos, photos_path = crop_face.main(secure_path, img_size=DATASET_IMG_SIZE)

    static_faces = []
    for face_path in photos_path:
        filename = os.path.basename(face_path)
        static_img_path = 'static/img/cropped_{}'.format(filename)
        log.info('Moving %s to %s', face_path, static_img_path)
        os.rename(face_path, static_img_path)
        static_faces.append(static_img_path)

    log.info('Computing embeddings for photos')
    embeddings = []
    for photo in photos:
        log.info('Image: {}'.format(photo))
        log.info('Image shape: {}'.format(photo.shape))
        embedding = app.facenet.compute_embedding(photo)[0]
        log.info('Embedding: {}'.format(embedding))
        embeddings.append(embedding.tolist())

    knn_req = {'query': embeddings, 'K': 5, 'ef': 100}
    knn = requests.get('http://localhost:5000/knn', json=knn_req)
    neighbors = knn.json()
    log.info('Neighbors recieved: {}'.format(neighbors))

    neighbors_static = {}
    for neighbor in itertools.chain.from_iterable(neighbors):
        img_name = '{:06d}.jpg'.format(neighbor + 1)
        img_path = os.path.join(DATASET, img_name)
        static_img_path = 'static/img/neighbor_{}'.format(img_name)

        if not neighbor in neighbors_static:
            log.info('Copying %s to %s', img_path, static_img_path)
            shutil.copyfile(img_path, static_img_path)
            neighbors_static[neighbor] = static_img_path

    return render_template(
        'results.html',
        images=[
            {
                'name': os.path.basename(face_path),
                'path': '/{}'.format(face_path),
                'neighbors': [neighbors_static[n] for n in neighbors[idx]],
            }
            for idx, face_path in enumerate(static_faces)
        ]
    )


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)

