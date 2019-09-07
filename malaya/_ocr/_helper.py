import numpy as np
from ._feature import histogram_norm

SMALL_HEIGHT = 800


def resize(img, height = SMALL_HEIGHT, allways = False):
    import cv2

    if img.shape[0] > height or allways:
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))

    return img


def ratio(img, height = SMALL_HEIGHT):
    return img.shape[0] / height


def edges_det(img, min_val, max_val):
    import cv2

    img = cv2.cvtColor(resize(img), cv2.COLOR_BGR2GRAY)

    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4
    )
    img = cv2.medianBlur(img, 11)
    img = cv2.copyMakeBorder(
        img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value = [0, 0, 0]
    )
    return cv2.Canny(img, min_val, max_val)


def four_corners_sort(pts):
    diff = np.diff(pts, axis = 1)
    summ = pts.sum(axis = 1)
    return np.array(
        [
            pts[np.argmin(summ)],
            pts[np.argmax(diff)],
            pts[np.argmax(summ)],
            pts[np.argmin(diff)],
        ]
    )


def contour_offset(cnt, offset):
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt


def find_page_contours(edges, img):
    import cv2

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_COUNTOUR_AREA = height * width * 0.5
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    max_area = MIN_COUNTOUR_AREA
    page_contour = np.array(
        [[0, 0], [0, height - 5], [width - 5, height - 5], [width - 5, 0]]
    )

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        if (
            len(approx) == 4
            and cv2.isContourConvex(approx)
            and max_area < cv2.contourArea(approx) < MAX_COUNTOUR_AREA
        ):

            max_area = cv2.contourArea(approx)
            page_contour = approx[:, 0]
    page_contour = four_corners_sort(page_contour)
    return contour_offset(page_contour, (-5, -5))


def persp_transform(img, s_points):
    import cv2

    height = max(
        np.linalg.norm(s_points[0] - s_points[1]),
        np.linalg.norm(s_points[2] - s_points[3]),
    )
    width = max(
        np.linalg.norm(s_points[1] - s_points[2]),
        np.linalg.norm(s_points[3] - s_points[0]),
    )

    t_points = np.array(
        [[0, 0], [0, height], [width, height], [width, 0]], np.float32
    )

    if s_points.dtype != np.float32:
        s_points = s_points.astype(np.float32)

    M = cv2.getPerspectiveTransform(s_points, t_points)
    return cv2.warpPerspective(img, M, (int(width), int(height)))


def sobel(channel):
    import cv2

    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)


def edge_detect(im):
    return np.max(
        np.array([sobel(im[:, :, 0]), sobel(im[:, :, 1]), sobel(im[:, :, 2])]),
        axis = 0,
    )


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return [x, y, w, h]


def intersect(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return False
    return True


def group_rectangles(rec):
    tested = [False for i in range(len(rec))]
    final = []
    i = 0
    while i < len(rec):
        if not tested[i]:
            j = i + 1
            while j < len(rec):
                if not tested[j] and intersect(rec[i], rec[j]):
                    rec[i] = union(rec[i], rec[j])
                    tested[j] = True
                    j = i
                j += 1
            final += [rec[i]]
        i += 1

    return final


def text_detect(img, original):
    import cv2

    small = resize(img, 2000)
    image = resize(original, 2000)
    cp_image = image.copy()
    mask = np.zeros(small.shape, np.uint8)
    cnt, hierarchy = cv2.findContours(
        np.copy(small), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    index = 0
    boxes = []
    while index >= 0:
        x, y, w, h = cv2.boundingRect(cnt[index])
        cv2.drawContours(mask, cnt, index, (255, 255, 255), cv2.FILLED)
        maskROI = mask[y : y + h, x : x + w]
        r = cv2.countNonZero(maskROI) / (w * h)
        if (
            r > 0.1
            and 2000 > w > 10
            and 1600 > h > 10
            and h / w < 3
            and w / h < 10
        ):
            boxes += [[x, y, w, h]]

        index = hierarchy[0][index][0]

    bounding_boxes = np.array([0, 0, 0, 0])
    for (x, y, w, h) in boxes:
        cv2.rectangle(cp_image, (x, y), (x + w, y + h), (0, 255, 0), 8)
        bounding_boxes = np.vstack(
            (bounding_boxes, np.array([x, y, x + w, y + h]))
        )

    boxes = bounding_boxes.dot(ratio(image, small.shape[0])).astype(np.int64)
    return boxes[1:]


def sort_words(box):
    boxes = box.copy()
    """Sort boxes - (x, y, x+w, y+h) from left to right, top to bottom."""
    mean_height = sum([y2 - y1 for _, y1, _, y2 in boxes]) / len(boxes)

    boxes.view('i8,i8,i8,i8').sort(order = ['f1'], axis = 0)
    current_line = boxes[0][1]
    lines = []
    tmp_line = []
    for box in boxes:
        if box[1] > current_line + mean_height:
            lines.append(tmp_line)
            tmp_line = [box]
            current_line = box[1]
            continue
        tmp_line.append(box)
    lines.append(tmp_line)

    for line in lines:
        line.sort(key = lambda box: box[0])

    return lines


def detect_page(image, min_val = 200, max_val = 250):
    import cv2

    edges_image = edges_det(image, min_val, max_val)
    edges_image = cv2.morphologyEx(
        edges_image, cv2.MORPH_CLOSE, np.ones((5, 11))
    )
    page_contour = find_page_contours(edges_image, resize(image))
    page_contour = page_contour.dot(ratio(image))
    return persp_transform(image, page_contour)


def detect_bounding_boxes(image, y_sensitivity = 5, x_sensitivity = 10):
    import cv2

    edges = edge_detect(image)
    ret, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    bw_image = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((y_sensitivity, x_sensitivity), np.uint8),
    )
    boxes = text_detect(bw_image, image)
    return boxes


def get_image(image):
    import cv2

    if isinstance(image, np.ndarray):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if isinstance(image, str):
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)

    return histogram_norm(image)


def get_image_color(image):
    import cv2

    if isinstance(image, np.ndarray):
        if len(image.shape) > 2:
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2BGR)
    if isinstance(image, str):
        image = cv2.imread(image)

    return image


def feature_ocr(img, image_height = 60, image_width = 240, image_channel = 1):
    from skimage.transform import resize as imresize
    import cv2

    im = imresize(
        cv2.flip((img.astype(np.float32) / 255.0), 1),
        (image_height, image_width, image_channel),
    )[:, :, 0]
    return im
