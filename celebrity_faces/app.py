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

# in other order raises ImportError: dlopen: cannot load any more object with static TLS
from utils.face_detection import crop_face

# Decoder, LinearBnRelu must be imported!
from utils.gan.model import load_model as load_gan_model, decode_pairs, Decoder, LinearBnRelu
from utils.facemodel import model as facenet_model


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.facenet = facenet_model.Facenet('utils/facemodel/facenet_models/20170512-110547')
app.gan = load_gan_model('utils/gan/decoder_data')

DATASET_IMG_SIZE = (160, 160)
DATASET = 'img_align_celeba'

if not os.path.exists('uploads'):
    os.mkdir('uploads')


def validate_img():
    if 'user_image' not in request.files:
        return None, None, redirect(request.url)

    user_image = request.files['user_image']
    secure_path = os.path.join('uploads', secure_filename(user_image.filename))

    if not secure_path.endswith('jpg') and not secure_path.endswith('jpeg') and not secure_path.endswith('png'):
        return None, None, render_template('error.html', error='Image must be .jpg/.jpeg or .png')

    return user_image.read(), secure_path, None


def save_img(data, path):
    log.info('Saving user image to %s', path)
    npimg = np.fromstring(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    log.info('Image: {}'.format(img))
    misc.imsave(path, img)


def move_to_static(photos_pathes):
    static_faces = []
    for face_path in photos_pathes:
        static_img_path = 'static/img/cropped_{}'.format(os.path.basename(face_path))
        log.info('Moving %s to %s', face_path, static_img_path)
        os.rename(face_path, static_img_path)
        static_faces.append(static_img_path)
    return static_faces


def compute_embeddings(photos):
    log.info('Computing embeddings for photos')
    embeddings = []
    for photo in photos:
        log.info('Image: {}'.format(photo))
        log.info('Image shape: {}'.format(photo.shape))
        embedding = app.facenet.compute_embedding(photo)[0]
        log.info('Embedding: {}'.format(embedding))
        embeddings.append(embedding.tolist())
    return embeddings


def get_neighbors(embeddings, K, ef):
    knn_req = {'query': embeddings, 'K': K, 'ef': ef}
    knn = requests.get('http://localhost:5000/knn', json=knn_req)
    neighbors = knn.json()
    log.info('Neighbors recieved: {}'.format(neighbors))
    return neighbors


def copy_neighbors_to_static(neighbors):
    neighbors_static = {}
    for neighbor in itertools.chain.from_iterable(neighbors):
        img_name = '{:06d}.jpg'.format(neighbor + 1)
        img_path = os.path.join(DATASET, img_name)
        static_img_path = 'static/img/neighbor_{}'.format(img_name)

        if not neighbor in neighbors_static:
            log.info('Copying %s to %s', img_path, static_img_path)
            shutil.copyfile(img_path, static_img_path)
            neighbors_static[neighbor] = static_img_path

    return neighbors_static


def compute_gan_imgs(embeddings, neighbors):
    neighbors_idxs = [n[0] for n in neighbors]
    return decode_pairs(app.gan, list(zip(embeddings, neighbors_idxs)))


def save_imgs_to_static(imgs, prefix):
    names = []

    for idx, img in enumerate(imgs):
        path = 'static/img/{}_{}.jpg'.format(prefix, idx)
        log.info('Saving image to {}'.format(path))
        misc.imsave(path, img)
        names.append(path)

    return names


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('base.html')

    user_img, secure_path, error = validate_img()
    if error:
        return error

    save_img(user_img, secure_path)

    photos, photos_path = crop_face.main(secure_path, img_size=DATASET_IMG_SIZE)
    static_faces = move_to_static(photos_path)

    embeddings = compute_embeddings(photos)
    neighbors = get_neighbors(embeddings, K=5, ef=100)

    neighbors_static = copy_neighbors_to_static(neighbors)

    gan_imgs = compute_gan_imgs(embeddings, neighbors)
    static_gans = save_imgs_to_static(gan_imgs, prefix='gan')

    return render_template(
        'results.html',
        images=[
            {
                'name': os.path.basename(face_path),
                'path': '/{}'.format(face_path),
                'neighbors': [neighbors_static[n] for n in neighbors[idx]],
                'gan': static_gans[idx],
            }
            for idx, face_path in enumerate(static_faces)
        ]
    )


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)

