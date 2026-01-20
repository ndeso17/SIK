from .utils_association import associate_driver_to_vehicle

class DriverAttribution:

    def __init__(self):
        pass

    def get_driver(self, vehicle_bbox, vehicle_type, persons, person_bboxes, plate_bbox=None):
        if not persons:
            return None
        (vx1, vy1, vx2, vy2) = vehicle_bbox
        vw = vx2 - vx1
        vh = vy2 - vy1
        if vehicle_type == 'motorcycle':
            cx1 = vx1
            cx2 = vx2
            cx_top = vy1 - int(0.8 * vh)
            cx_bottom = vy2
            cabin = [int(cx1), int(cx_top), int(cx2), int(cx_bottom)]
        elif plate_bbox is not None:
            (px1, py1, px2, py2) = plate_bbox
            cx1 = vx1
            cx2 = vx2
            cx_top = vy1
            cx_bottom = max(vy1, py1)
            cabin = [int(cx1), int(cx_top), int(cx2), int(cx_bottom)]
        elif vehicle_type == 'car':
            cx1 = vx1
            cx2 = vx1 + int(0.6 * vw)
            cx_top = vy1
            cx_bottom = vy1 + int(0.6 * vh)
            cabin = [int(cx1), int(cx_top), int(cx2), int(cx_bottom)]
        else:
            cx1 = vx1
            cx2 = vx2
            cx_top = vy1
            cx_bottom = vy1 + int(0.6 * vh)
            cabin = [int(cx1), int(cx_top), int(cx2), int(cx_bottom)]
        from .utils_bbox import calculate_iou_over_a, get_center_point
        best_score = -1
        best_idx = -1
        best_reason = ''
        vehicle_area = vw * vh
        for (i, person) in enumerate(persons):
            pbox = person.get('bbox') if 'bbox' in person else person_bboxes[i]
            (px1, py1, px2, py2) = pbox
            pw = px2 - px1
            ph = py2 - py1
            person_area = pw * ph
            if vehicle_area > 0:
                ratio = person_area / vehicle_area
                if ratio < 0.15:
                    continue
            (cx, cy) = get_center_point(pbox)
            center_inside = cabin[0] <= cx <= cabin[2] and cabin[1] <= cy <= cabin[3]
            overlap = calculate_iou_over_a(pbox, cabin)
            score = 0.0
            reason = ''
            if center_inside:
                score = 1.0 + overlap
                reason = 'center_in_cabin'
            elif overlap > 0.3:
                score = overlap
                reason = 'overlap_gt_30pct'
            if score > best_score:
                best_score = score
                best_idx = i
                best_reason = reason
        if best_idx != -1 and best_score > 0:
            driver = persons[best_idx].copy()
            driver_bbox = persons[best_idx].get('bbox') if 'bbox' in persons[best_idx] else person_bboxes[best_idx]
            return {'person_id': f'p{best_idx + 1}', 'bbox': {'x': int(driver_bbox[0]), 'y': int(driver_bbox[1]), 'w': int(driver_bbox[2] - driver_bbox[0]), 'h': int(driver_bbox[3] - driver_bbox[1])}, 'confidence_reason': best_reason}
        return None