import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET


# returns the face of a single image
def get_face(img, xmin, ymin, xmax, ymax):
    return img[ymin:ymax,xmin:xmax]


# loads data into memory given full path
def load_data_to_memory(path):

    items = [i[:-4] for i in os.listdir(os.path.join(path, 'img'))]
    annotate = [ET.parse(os.path.join(path,'annotations') + '/' + i + '.xml').getroot() for i in items]
    images = [cv2.imread(os.path.join(path,'img') + '/' + i + '.jpg') for i in items]
    n_of_obj_in_img = [len(root) - 5 for root in annotate]

    images = dict(zip(items,images))
    annotate = dict(zip(items, annotate))
    n_of_obj_in_img = dict(zip(items, n_of_obj_in_img))

    items_ = list()
    faces_ = list()
    images_ = list()
    emotion = list()
    bounding = list()

    for item in items:
        for n in (range(n_of_obj_in_img[item])):
            items_.append(item)
            images_.append(images[item])
            # parse xml tree to extract the emotion
            root = annotate[item]
            emotion.append(root[5+n][0].text)
            bounding.append((int(root[5+n][4][0].text),
              int(root[5+n][4][1].text),
              int(root[5+n][4][2].text),
              int(root[5+n][4][3].text)))
            faces_.append(get_face(images[item], int(root[5+n][4][0].text),
              int(root[5+n][4][1].text),
              int(root[5+n][4][2].text),
              int(root[5+n][4][3].text)))


    # parse xml tree to extract the bounding box (xmin, ymin, xmax, ymax format)
    data = list(zip(items_, faces_, images_, emotion, bounding))

    return data


def preprocess(img, size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR) # shrinking images -> inter_linear
    norm_img = np.zeros((size, size))
    norm_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    return norm_img

