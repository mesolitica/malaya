import numpy as np


def bilateral_norm(img):
    import cv2

    img = cv2.bilateralFilter(img, 9, 15, 30)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def histogram_norm(img):
    import cv2

    img = bilateral_norm(img)
    threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
        1
    ]
    unique, counts = np.unique(threshold, return_counts = True)
    dict_color = dict(zip(unique, counts))
    if dict_color[0] > dict_color[255]:
        minus = 0
    else:
        minus = 255
    add_img = minus - threshold
    return add_img


def elastic_transform(image, alpha, sigma, alpha_affine, random_state = None):
    from scipy.ndimage.interpolation import map_coordinates
    from scipy.ndimage.filters import gaussian_filter
    import cv2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    blur_size = int(4 * sigma) | 1
    dx = alpha * cv2.GaussianBlur(
        (random_state.rand(*shape) * 2 - 1),
        ksize = (blur_size, blur_size),
        sigmaX = sigma,
    )
    dy = alpha * cv2.GaussianBlur(
        (random_state.rand(*shape) * 2 - 1),
        ksize = (blur_size, blur_size),
        sigmaX = sigma,
    )

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    image = map_coordinates(
        image, indices, order = 1, mode = 'constant'
    ).reshape(shape)

    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 4
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + random_state.uniform(
        -alpha_affine, alpha_affine, size = pts1.shape
    ).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode = cv2.BORDER_CONSTANT
    )

    return image


def get_hog_features(
    img, orient, pix_per_cell, cell_per_block, feature_vec = True
):
    from skimage.feature import hog

    features = hog(
        img,
        orientations = orient,
        pixels_per_cell = (pix_per_cell, pix_per_cell),
        cells_per_block = (cell_per_block, cell_per_block),
        transform_sqrt = True,
        visualise = False,
        feature_vector = feature_vec,
    )
    return features


def bin_spatial(img, size = (16, 16)):
    import cv2

    return cv2.resize(img, size).ravel()


def img_features(
    feature_image, hist_bins, orient, pix_per_cell, cell_per_block, spatial_size
):
    features = []
    features.append(bin_spatial(feature_image, size = spatial_size))
    features.append(
        get_hog_features(feature_image, orient, pix_per_cell, cell_per_block)
    )
    return features


def extract_features(img):
    import cv2

    img = cv2.resize(img, (350, 125))
    file_features = img_features(
        img,
        hist_bins = 32,
        orient = 8,
        pix_per_cell = 8,
        cell_per_block = 2,
        spatial_size = (32, 32),
    )
    return np.concatenate(file_features)
