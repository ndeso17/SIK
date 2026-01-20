from .utils_bbox import calculate_iou_over_a, get_center_point, is_point_inside_box

def associate_vehicle_to_plate(vehicle_bboxes, plate_bbox):
    best_iou = 0
    best_vehicle_idx = -1
    for (i, vehicle_bbox) in enumerate(vehicle_bboxes):
        iou = calculate_iou_over_a(plate_bbox, vehicle_bbox)
        if iou > 0.05 and iou > best_iou:
            best_iou = iou
            best_vehicle_idx = i
    return best_vehicle_idx

def associate_driver_to_vehicle(vehicle_bbox, vehicle_type, person_bboxes):
    best_iou = 0
    best_person_idx = -1
    (vx1, vy1, vx2, vy2) = vehicle_bbox
    vw = vx2 - vx1
    vh = vy2 - vy1
    if vehicle_type == 'motorcycle':
        driver_roi = [vx1, int(vy1 - vh * 0.5), vx2, int(vy1 + vh * 0.8)]
    elif vehicle_type == 'car':
        driver_roi = [vx1, vy1, vx2, int(vy1 + vh * 0.6)]
    else:
        driver_roi = vehicle_bbox
    from .utils_bbox import calculate_iou_over_a
    for (i, person_bbox) in enumerate(person_bboxes):
        iou = calculate_iou_over_a(person_bbox, driver_roi)
        if iou > 0.3 and iou > best_iou:
            best_iou = iou
            best_person_idx = i
    return best_person_idx