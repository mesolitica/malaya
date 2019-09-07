from sklearn.linear_model import LogisticRegression
from ._feature import histogram_norm, extract_features
from ._helper import (
    get_image,
    detect_page,
    detect_bounding_boxes,
    sort_words,
    get_image_color,
)
from ._encode import encode_map, decode_map
import numpy as np


class DETECT_JAWI:
    def __init__(self, logistic):
        self._logistic = logistic
        self._label = ['not jawi', 'jawi']

    def _predict(self, imgs):
        features = [extract_features(im) for im in imgs]
        return self._logistic.predict_proba(features)

    def predict(self, image, get_proba = False):
        """
        classify an image.

        Parameters
        ----------
        image : str / np.array
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        dictionary: results
        """
        if not isinstance(image, str) and not isinstance(image, np.ndarray):
            raise ValueError('image must be a string or a numpy array')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')

        image = get_image(image)
        result = self._predict([image])

        result = result[0]
        if get_proba:
            return {self._label[i]: result[i] for i in range(len(result))}
        else:
            return self._label[np.argmax(result)]

    def predict_batch(self, images, get_proba = False):
        """
        classify list of images.

        Parameters
        ----------
        strings : list of strings / list of np.array
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(images, list):
            raise ValueError('input must be a list')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        if not isinstance(images[0], str) and not isinstance(
            images[0], np.ndarray
        ):
            raise ValueError('elements must be a string or a numpy array')

        images = [get_image(image) for image in images]
        results = self._predict(images)

        if get_proba:
            outputs = []
            for result in results:
                outputs.append(
                    {self._label[i]: result[i] for i in range(len(result))}
                )
            return outputs
        else:
            return [
                self._label[result] for result in np.argmax(results, axis = 1)
            ]


class JAWI_TO_MALAY:
    def __init__(self, X, logits, sess, jawi_detector = None):
        self._X = X
        self._logits = logits
        self._sess = sess
        self._jawi_detector = jawi_detector

    def detect(
        self, image, detect_page = True, y_sensitivity = 5, x_sensitivity = 10
    ):
        """
        do OCR for an image.

        Parameters
        ----------
        image : str / np.array
        detect_page: bool, optional (default=False)
            If True, it will transform the image to detect 4 edges like a paper.
        y_sensitivity: int, optional (default=5)
            pixel sensitivity in term of y-axis.
        x_sensitivity: int, optional (default=10)
            pixel sensitivity in term of x-axis.

        Returns
        -------
        dictionary: results
        """
        if not isinstance(image, str) and not isinstance(image, np.ndarray):
            raise ValueError('image must be a string or a numpy array')
        if not isinstance(detect_page, bool):
            raise ValueError('detect_page must be a boolean')
        if not isinstance(y_sensitivity, int):
            raise ValueError('y_sensitivity must be an integer')
        if not isinstance(x_sensitivity, int):
            raise ValueError('x_sensitivity must be an integer')
        image = get_image_color(image)
        if detect_page:
            image = detect_page(image)
        boxes = detect_bounding_boxes(
            image, y_sensitivity = y_sensitivity, x_sensitivity = x_sensitivity
        )
        cp_image = image.copy()
        if self._jawi_detector:
            imgs = []
            for x1, y1, x2, y2 in boxes:
                imgs.append(image[y1:y2, x1:x2])
            detected = detector.predict_batch(imgs)
        else:
            detected = ['jawi'] * len(boxes)

        new_boxes = []
        for (no, (x, y, w, h)) in enumerate(boxes):
            if detected[no] == 'jawi':
                new_boxes.append(boxes[no])
                c = (0, 255, 0)
            else:
                c = (255, 0, 0)
            cv2.rectangle(cp_image, (x, y), (w, h), c, 8)
        new_boxes = np.array(new_boxes)
        sorted_boxes = sort_words(new_boxes)
        strings = []
        for boxes in sorted_boxes:
            imgs = []
            for x1, y1, x2, y2 in boxes[::-1]:
                img = histogram_norm(image[y1:y2, x1:x2, 0])
            batch_x = np.expand_dims(np.array(imgs), -1)
            decoded = self._sess.run(self._logits, feed_dict = {self._X: img})[
                :, :, 0
            ]
