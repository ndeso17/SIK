from flask import Blueprint, request, render_template, jsonify, Response
from Controllers.api_controller import ApiController
from Models import db, VehicleIdentity, VehicleObservation
from Libs.config import VEHICLE_UI_PER_PAGE
api = Blueprint('api', __name__)

@api.route('/', methods=['GET'])
def index():
    from collections import Counter
    from sqlalchemy import func
    type_counter = Counter()
    color_counter = Counter()
    letter_counter = Counter()
    total_detections = 0
    identities = VehicleIdentity.query.all()
    for identity in identities:
        if identity.vehicle_type:
            t_norm = identity.vehicle_type.strip().title()
            type_counter[t_norm] += identity.detection_count or 1
        if identity.vehicle_color:
            c_norm = identity.vehicle_color.strip().title()
            color_counter[c_norm] += identity.detection_count or 1
        import re
        if identity.plate_text:
            match = re.match('^([A-Z]+)', identity.plate_text.upper())
            if match:
                code = match.group(1)
                letter_counter[code] += 1
        total_detections += identity.detection_count or 1
    type_stats = [{'label': k, 'count': v} for (k, v) in type_counter.most_common()]
    color_stats = [{'label': k, 'count': v} for (k, v) in color_counter.most_common()]
    letter_stats = [{'label': k, 'count': v} for (k, v) in sorted(letter_counter.items(), key=lambda x: (-x[1], x[0]))]
    total_vehicles = len(identities)
    return render_template('index.html', type_stats=type_stats, color_stats=color_stats, letter_stats=letter_stats, total_vehicles=total_vehicles, total_detections=total_detections)

@api.route('/video_ui', methods=['GET'])
def video_ui():
    source = request.args.get('source', 'webcam')
    url = request.args.get('url', '')
    return render_template('video.html', source=source, url=url)

@api.route('/video_upload', methods=['GET'])
def video_upload():
    return render_template('upload_video.html')

@api.route('/api/video', methods=['POST'])
def process_video():
    from flask import flash
    from Controllers.api_controller import ApiController
    return ApiController.upload_video(request)

@api.route('/upload_image', methods=['GET'])
def upload_image_ui():
    return render_template('upload_image.html')

@api.route('/api/image', methods=['POST'])
def process_image():
    return ApiController.upload_image(request)

@api.route('/api/process_frame_client', methods=['POST'])
def process_frame_client():
    return ApiController.process_frame_client(request)

@api.route('/api/webcam')
def webcam_feed():
    device = request.args.get('device', default=0, type=int)
    fps = request.args.get('fps', default=None, type=float)
    return ApiController.video_feed_webcam(device=device, target_fps=fps)

@api.route('/api/ipcam')
def ipcam_feed():
    url = request.args.get('url')
    if not url:
        return ('URL hilang', 400)
    fps = request.args.get('fps', default=None, type=float)
    return ApiController.video_feed_ipcam(url, target_fps=fps)

@api.route('/api/stream_status')
def stream_status():
    return jsonify({'success': True, 'data': ApiController.get_stream_status()})

@api.route('/api/recent_observations')
def recent_observations():
    data = ApiController.get_recent_detections()
    return jsonify({'success': True, 'data': data})

@api.route('/api/confirm_observation/<detection_id>', methods=['POST'])
def confirm_observation(detection_id):
    (success, msg) = ApiController.confirm_detection(detection_id)
    return jsonify({'success': success, 'message': msg})

@api.route('/clusters')
def clusters_ui():
    from flask import redirect, url_for
    return redirect(url_for('admin.vehicles_list'))

@api.route('/clusters/<cluster_id>')
def cluster_detail(cluster_id):
    from flask import redirect, url_for
    try:
        identity_id = int(cluster_id)
        return redirect(url_for('admin.vehicle_detail', id=identity_id))
    except ValueError:
        identity = VehicleIdentity.query.filter_by(plate_text=cluster_id).first()
        if identity:
            return redirect(url_for('admin.vehicle_detail', id=identity.id))
        return ('Tidak ditemukan', 404)

@api.route('/video_result/<path:filename>')
def serve_video_result(filename):
    from flask import send_from_directory
    import os
    directory = os.path.abspath(os.path.join(os.getcwd(), 'results', 'processed_videos'))
    return send_from_directory(directory, filename)