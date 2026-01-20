def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - area_inter
    if union == 0:
        return 0
    return area_inter / union

def calculate_iou_over_a(boxA, boxB):
    x1_inter = max(boxA[0], boxB[0])
    y1_inter = max(boxA[1], boxB[1])
    x2_inter = min(boxA[2], boxB[2])
    y2_inter = min(boxA[3], boxB[3])
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter
    area_a = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    if area_a == 0:
        return 0
    return area_inter / area_a

def get_center_point(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def is_point_inside_box(point, box):
    (x, y) = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]

def xywh_to_xyxy(x, y, w, h):
    return [x, y, x + w, y + h]

def scale_coords(img_shape, coords, img0_shape):
    pass