import os
import numpy as np
import tensorflow as tf

from scipy import misc

from utils.facemodel import facenet


class Facenet(object):
    img_size = (160, 160, 3)
    emb_size = 128

    def __init__(self, model_path):
        self.sess = tf.InteractiveSession()
        self.load_model(model_path)

    def __setattr__(self, name, value):
        if name in ("img_size", "emb_size"):
            raise AttributeError("Cannot modify .{}".format(name))
        else:
            object.__setattr__(self, name, value)

    def load_model(self, model_path):
        facenet.load_model(model_path, self.sess)

    def prewhiten(self, img):
        return facenet.prewhiten(img)

    def compute_embedding(self, img, do_prewhiten=True):
        if len(img.shape) not in (3, 4):
            raise ValueError("Expected 3 or 4 dimensions, got {}".format(len(img.shape)))
        if img.shape[-3:] != self.img_size:
            raise ValueError("Expected {} size image, got {}".format(self.img_size, img.shape))

        images = np.copy(img) if len(img.shape) == 4 else np.copy(img)[np.newaxis, :, :, :]
        if do_prewhiten:
            for i in range(images.shape[0]):
                images[i] = self.prewhiten(images[i])

        images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
        phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")
        embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")

        feed_dict = {
            images_placeholder: images,
            phase_train_placeholder: False
        }
        emb_array = self.sess.run(embeddings, feed_dict=feed_dict)

        return emb_array


def crop_image(img, height=160, width=160):
    h, w, c = img.shape
    h_pad = int((h - height) / 2)
    w_pad = int((w - width) / 2)
    return img[h_pad:-h_pad, w_pad:-w_pad, :]


def crop_images(img_dir_in, img_dir_out):
    if not os.path.exists(img_dir_in):
        raise FileNotFoundError('{} folder is not found'.format(img_dir_in))
    filenames = os.listdir(img_dir_in)

    if not os.path.exists(img_dir_out):
        os.mkdir(img_dir_out)
    for filename in filenames:
        img = misc.imread(os.path.join(img_dir_in, filename))
        cropped_img = crop_image(img)
        misc.imsave(os.path.join(img_dir_out, filename), cropped_img)


def read_images(img_paths):
    img_size = misc.imread(img_paths[0]).shape
    images = np.zeros(shape=(len(img_paths), *img_size))

    for i, img_path in enumerate(img_paths):
        images[i]  = misc.imread(img_path).atype(float)
    return images

# для построения эмбеддингов всего датасета
def get_batch(img_dir, batch_size, offset):
    filenames = sorted(os.listdir(img_dir))[offset:offset+batch_size]
    img_paths = [os.path.join(img_dir, filename) for filename in filenames]

    batch = read_images(img_paths)

    return batch

