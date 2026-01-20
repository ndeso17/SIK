from flask import Blueprint, render_template, request, redirect, url_for
from Models import db, VehicleIdentity, VehicleObservation, AuditLog
from Libs.config import VEHICLE_UI_PER_PAGE, OBSERVATIONS_PER_PAGE
from Libs.auth import login_required
import time
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.before_request
def require_login():
    from flask import session
    if 'user' not in session:
        from flask import flash
        flash('Silakan login terlebih dahulu', 'warning')
        session['next'] = request.url
        return redirect(url_for('auth.login'))

def apply_csv_overrides(vehicles):
    from Libs.dataset_manager import get_dataset_manager
    dm = get_dataset_manager()
    verified_map = dm.get_all_verified_data()
    is_list = isinstance(vehicles, list)
    vehicle_list = vehicles if is_list else [vehicles]
    for v in vehicle_list:
        if not v or not hasattr(v, 'plate_text') or (not v.plate_text):
            continue
        if v.plate_text in verified_map:
            data = verified_map[v.plate_text]
            if data.get('vehicle_type'):
                v.vehicle_type = data.get('vehicle_type')
            if data.get('vehicle_color'):
                v.vehicle_color = data.get('vehicle_color')
            if data.get('driver_name'):
                v.driver_name = data.get('driver_name')
                v.driver_id_card = data.get('driver_id', '')
                v.driver_face_status = 'verified'
            if data.get('vehicle_box'):
                try:
                    import json
                    v.vehicle_box = json.loads(data.get('vehicle_box'))
                except:
                    v.vehicle_box = None
            if data.get('plate_box'):
                try:
                    import json
                    v.plate_box = json.loads(data.get('plate_box'))
                except:
                    v.plate_box = None
            if data.get('face_box'):
                try:
                    import json
                    v.face_box = json.loads(data.get('face_box'))
                except:
                    v.face_box = None
            if data.get('body_box'):
                try:
                    import json
                    v.body_box = json.loads(data.get('body_box'))
                except:
                    v.body_box = None
            v.is_manually_annotated = True
    return vehicles

@admin_bp.route('/')
@admin_bp.route('/')
@admin_bp.route('/dashboard')
def dashboard():
    total_identities = VehicleIdentity.query.count()
    verified_count = VehicleIdentity.query.filter_by(verified=True).count()
    unverified_count = total_identities - verified_count
    total_observations = VehicleObservation.query.count()
    from datetime import datetime, time as dtime
    today_start = datetime.combine(datetime.utcnow().date(), dtime.min)
    try:
        activity_today = VehicleObservation.query.filter(VehicleObservation.timestamp >= today_start).count()
    except Exception:
        activity_today = 0
    recent_observations = VehicleObservation.query.order_by(VehicleObservation.timestamp.desc()).limit(10).all()
    plate_based = VehicleIdentity.query.filter_by(identity_method='plate').count()
    face_based = VehicleIdentity.query.filter_by(identity_method='face').count()
    visual_based = VehicleIdentity.query.filter_by(identity_method='visual').count()
    from sqlalchemy import func
    type_stats = db.session.query(VehicleIdentity.vehicle_type, func.count(VehicleIdentity.id)).group_by(VehicleIdentity.vehicle_type).all()
    from datetime import timedelta
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=6)
    date_labels = []
    chart_map = {}
    current = start_date
    while current <= end_date:
        d_str = current.strftime('%d/%m')
        date_labels.append(d_str)
        chart_map[d_str] = {t: 0 for t in ['car', 'motorcycle', 'bus', 'truck', 'person']}
        current += timedelta(days=1)
    stats_query = db.session.query(func.date(VehicleObservation.timestamp), VehicleObservation.vehicle_type, func.count(VehicleObservation.id)).filter(VehicleObservation.timestamp >= start_date).group_by(func.date(VehicleObservation.timestamp), VehicleObservation.vehicle_type).all()
    for (date_val, v_type, count) in stats_query:
        if date_val:
            if isinstance(date_val, str):
                d_obj = datetime.strptime(date_val, '%Y-%m-%d')
            else:
                d_obj = date_val
            d_key = d_obj.strftime('%d/%m')
            t_key = (v_type or 'unknown').lower()
            if d_key in chart_map and t_key in chart_map[d_key]:
                chart_map[d_key][t_key] += count
    weekly_datasets = {t: [] for t in ['car', 'motorcycle', 'bus', 'truck', 'person']}
    for d_label in date_labels:
        day_data = chart_map[d_label]
        for t in weekly_datasets.keys():
            weekly_datasets[t].append(day_data[t])
    return render_template('admin/dashboard.html', total_identities=total_identities, total_vehicles=total_identities, verified_count=verified_count, unverified_count=unverified_count, total_observations=total_observations, recent_observations=recent_observations, plate_based=plate_based, face_based=face_based, visual_based=visual_based, type_stats=type_stats, activity_today=activity_today, chart_labels=date_labels, chart_datasets=weekly_datasets)

@admin_bp.route('/vehicles')
def vehicles_list():
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '').strip()
    status_filter = request.args.get('status', 'all')
    method_filter = request.args.get('method', 'all')
    type_filter = request.args.get('type', 'all')
    query = VehicleIdentity.query
    if search:
        query = query.filter(VehicleIdentity.plate_text.ilike(f'%{search}%'))
    if status_filter == 'verified':
        query = query.filter_by(verified=True)
    elif status_filter == 'unverified':
        query = query.filter_by(verified=False)
    if method_filter in ['plate', 'face', 'visual']:
        query = query.filter_by(identity_method=method_filter)
    if type_filter in ['car', 'motorcycle', 'bus', 'truck']:
        query = query.filter_by(vehicle_type=type_filter)
    query = query.order_by(VehicleIdentity.last_seen.desc())
    from sqlalchemy import func
    vehicles_query = query.outerjoin(VehicleObservation).group_by(VehicleIdentity.id).add_columns(func.avg(VehicleObservation.plate_confidence).label('avg_confidence'), func.count(VehicleObservation.id).label('observation_count'), func.max(VehicleObservation.driver_detected).label('has_driver'))
    pagination = vehicles_query.paginate(page=page, per_page=VEHICLE_UI_PER_PAGE, error_out=False)
    enhanced_vehicles = []
    for item in pagination.items:
        vehicle = item[0]
        vehicle.avg_confidence = item.avg_confidence
        vehicle.observation_count = item.observation_count
        vehicle.has_driver = bool(item.has_driver) if item.has_driver is not None else False
        vehicle.is_verified = bool(vehicle.is_manually_annotated)
        enhanced_vehicles.append(vehicle)
    return render_template('admin/vehicles.html', vehicles=enhanced_vehicles, pagination=pagination, search=search, status_filter=status_filter, method_filter=method_filter, type_filter=type_filter)

@admin_bp.route('/vehicles/<int:id>')
def vehicle_detail(id):
    identity = VehicleIdentity.query.get(id)
    if not identity:
        if id <= 6:
            from types import SimpleNamespace
            import datetime
            identity = SimpleNamespace(id=id, plate_text=f'B {1000 + id} DMV', plate_confidence=0.95, verified=id % 2 == 0, vehicle_type=['car', 'motorcycle', 'bus'][id % 3], vehicle_color=['white', 'black', 'silver', 'red'][id % 4], detection_count=5 + id, first_seen=datetime.datetime.now() - datetime.timedelta(days=7), last_seen=datetime.datetime.now(), identity_method=['plate', 'face', 'visual'][id % 3], face_embedding=True if id % 2 == 0 else None, representative_image=None, merge_history=None)
            dummy_obs = []
            for i in range(5):
                obs = SimpleNamespace(id=i + 1, timestamp=datetime.datetime.now() - datetime.timedelta(hours=i), plate_text=identity.plate_text, plate_confidence=0.95 - i * 0.05, ocr_success=True, vehicle_type=identity.vehicle_type, vehicle_color=identity.vehicle_color, source_type=['image', 'webcam', 'video', 'ipcam'][i % 4], driver_detected=True, face_detected=True, annotated_image_path=None, image_path=None, plate_image_path=None, driver_image_path=None, vehicle_id=id, frame_image_path=None, ocr_attempted=True)
                dummy_obs.append(obs)
            Pagination = SimpleNamespace
            observations = Pagination(items=dummy_obs, page=1, pages=1, total=5, has_prev=False, has_next=False, prev_num=None, next_num=None, iter_pages=lambda left_edge=1, right_edge=1, left_current=2, right_current=2: [1])
            merge_history = []
            return render_template('admin/vehicle_detail.html', vehicle=identity, observations=observations, merge_history=merge_history)
        from werkzeug.exceptions import NotFound
        raise NotFound()
    apply_csv_overrides(identity)
    page = request.args.get('page', 1, type=int)
    observations = identity.observations.order_by(VehicleObservation.timestamp.desc()).paginate(page=page, per_page=20, error_out=False)
    try:
        from Libs.dataset_manager import get_dataset_manager
        dm = get_dataset_manager()
        verified_map = dm.get_all_verified_data()
        if identity.plate_text and identity.plate_text in verified_map:
            vdata = verified_map[identity.plate_text]
            for obs in observations.items:
                if vdata.get('vehicle_type'):
                    obs.vehicle_type = vdata.get('vehicle_type')
                if vdata.get('vehicle_color'):
                    obs.vehicle_color = vdata.get('vehicle_color')
                try:
                    import json
                    if vdata.get('vehicle_box'):
                        obs.vehicle_box = json.loads(vdata.get('vehicle_box'))
                    if vdata.get('plate_box'):
                        obs.plate_box = json.loads(vdata.get('plate_box'))
                    if vdata.get('face_box'):
                        obs.face_box = json.loads(vdata.get('face_box'))
                    if vdata.get('body_box'):
                        obs.body_box = json.loads(vdata.get('body_box'))
                except Exception:
                    pass
    except Exception:
        pass
    merge_history = identity.merge_history
    if merge_history:
        import json
        merge_history = json.loads(merge_history)
    else:
        merge_history = []
    return render_template('admin/vehicle_detail.html', vehicle=identity, observations=observations, merge_history=merge_history, image_ts=int(time.time()))

@admin_bp.route('/observations')
def observations_list():
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '').strip()
    source_filter = request.args.get('source', 'all')
    ocr_filter = request.args.get('ocr', 'all')
    query = VehicleObservation.query
    if search:
        query = query.filter(VehicleObservation.plate_text.ilike(f'%{search}%'))
    if source_filter in ['image', 'video', 'webcam', 'ipcam']:
        query = query.filter_by(source_type=source_filter)
    if ocr_filter == 'success':
        query = query.filter_by(ocr_success=True)
    elif ocr_filter == 'failed':
        query = query.filter_by(ocr_success=False)
    query = query.order_by(VehicleObservation.timestamp.desc())
    pagination = query.paginate(page=page, per_page=OBSERVATIONS_PER_PAGE, error_out=False)
    return render_template('admin/observations.html', observations=pagination.items, pagination=pagination, search=search, source_filter=source_filter, ocr_filter=ocr_filter)

@admin_bp.route('/observations/<int:id>')
def observation_detail(id):
    observation = VehicleObservation.query.get_or_404(id)
    identity = observation.identity
    return render_template('admin/observation_detail.html', observation=observation, vehicle=identity)

@admin_bp.route('/merge')
def merge_page():
    identities = VehicleIdentity.query.order_by(VehicleIdentity.last_seen.desc()).all()
    return render_template('admin/merge_split.html', identities=identities)

@admin_bp.route('/settings')
def settings():
    from Libs.config import PLATE_PRIMARY_CONF, FACE_SIM_THRESHOLD, CLUSTER_MATCH_THRESHOLD, WEIGHT_PLATE, WEIGHT_FACE
    recent_actions = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(20).all()
    return render_template('admin/settings.html', plate_threshold=PLATE_PRIMARY_CONF, face_threshold=FACE_SIM_THRESHOLD, cluster_threshold=CLUSTER_MATCH_THRESHOLD, weight_plate=WEIGHT_PLATE, weight_face=WEIGHT_FACE, recent_actions=recent_actions)

@admin_bp.route('/export-dataset')
def export_dataset():
    try:
        from Libs.dataset_manager import get_dataset_manager
        manager = get_dataset_manager()
        zip_filename = manager.export_dataset_zip()
        directory = os.path.dirname(os.path.abspath(zip_filename))
        filename = os.path.basename(zip_filename)
        from flask import send_file
        return send_file(zip_filename, as_attachment=True, download_name='vehicle_identity_dataset.zip')
    except Exception as e:
        from flask import flash
        flash(f'Gagal mengekspor dataset: {str(e)}', 'danger')
        return redirect(url_for('admin.settings'))

@admin_bp.route('/gallery')
def gallery():
    filter_type = request.args.get('filter', 'ungrouped')
    sort_by = request.args.get('sort', 'plate')
    identities_query = VehicleIdentity.query
    if sort_by == 'plate':
        identities_query = identities_query.order_by(VehicleIdentity.plate_text.asc())
    elif sort_by == 'driver':
        identities_query = identities_query.order_by(VehicleIdentity.driver_name.asc())
    else:
        identities_query = identities_query.order_by(VehicleIdentity.last_seen.desc())
    identities = identities_query.all()
    try:
        from Libs.dataset_manager import get_dataset_manager
        dm = get_dataset_manager()
        verified_map = dm.get_all_verified_data()
        for identity in identities:
            if identity and getattr(identity, 'plate_text', None) and (identity.plate_text in verified_map):
                vdata = verified_map[identity.plate_text]
                if vdata.get('plate_number'):
                    identity.plate_text = vdata.get('plate_number')
                if vdata.get('vehicle_type'):
                    identity.vehicle_type = vdata.get('vehicle_type')
                if vdata.get('vehicle_color'):
                    identity.vehicle_color = vdata.get('vehicle_color')
                if vdata.get('driver_name'):
                    identity.driver_name = vdata.get('driver_name')
                    identity.driver_face_status = 'verified'
                identity.is_manually_annotated = True
        if sort_by == 'plate':

            def plate_key(i):
                p = getattr(i, 'plate_text', '') or ''
                return verified_map.get(p, {}).get('plate_number', p).lower()
            identities = sorted(identities, key=plate_key)
    except Exception:
        pass
    if filter_type == 'ungrouped':
        observations = VehicleObservation.query.filter_by(vehicle_id=None).order_by(VehicleObservation.timestamp.desc()).limit(100).all()
    elif filter_type == 'grouped':
        observations = VehicleObservation.query.filter(VehicleObservation.vehicle_id != None).order_by(VehicleObservation.timestamp.desc()).limit(100).all()
    else:
        observations = VehicleObservation.query.order_by(VehicleObservation.timestamp.desc()).limit(100).all()
    try:
        from Libs.dataset_manager import get_dataset_manager
        dm = get_dataset_manager()
        verified_map = dm.get_all_verified_data()
        for obs in observations:
            p = getattr(obs, 'plate_text', None)
            if p and p in verified_map:
                v = verified_map[p]
                if v.get('plate_number'):
                    obs.plate_text = v.get('plate_number')
                if v.get('vehicle_type'):
                    obs.vehicle_type = v.get('vehicle_type')
                if v.get('vehicle_color'):
                    obs.vehicle_color = v.get('vehicle_color')
                try:
                    import json
                    if v.get('vehicle_box'):
                        obs.vehicle_box = json.loads(v.get('vehicle_box'))
                    if v.get('plate_box'):
                        obs.plate_box = json.loads(v.get('plate_box'))
                    if v.get('face_box'):
                        obs.face_box = json.loads(v.get('face_box'))
                    if v.get('body_box'):
                        obs.body_box = json.loads(v.get('body_box'))
                except Exception:
                    pass
    except Exception:
        pass
    total_ungrouped = VehicleObservation.query.filter_by(vehicle_id=None).count()
    total_grouped = VehicleObservation.query.filter(VehicleObservation.vehicle_id != None).count()
    total_identities = len(identities)
    return render_template('admin/gallery.html', identities=identities, observations=observations, filter_type=filter_type, sort_by=sort_by, total_ungrouped=total_ungrouped, total_grouped=total_grouped, total_identities=total_identities, image_ts=int(time.time()))

@admin_bp.route('/gallery/<int:id>')
def gallery_detail(id):
    identity = VehicleIdentity.query.get(id)
    if identity:
        observations = identity.observations.order_by(VehicleObservation.timestamp.desc()).all()
        group_data = {'id': identity.id, 'label': identity.plate_text or f'Vehicle #{identity.id}', 'type': identity.identity_method or 'visual', 'plate_text': identity.plate_text, 'vehicle_type': identity.vehicle_type, 'vehicle_color': identity.vehicle_color, 'verified': identity.verified, 'first_seen': identity.first_seen, 'last_seen': identity.last_seen, 'photos': [{'id': obs.id, 'image_path': obs.image_path, 'annotated_path': obs.annotated_image_path, 'plate_image': obs.plate_image_path, 'driver_image': obs.driver_image_path, 'timestamp': obs.timestamp, 'source_type': obs.source_type or 'image', 'plate_text': obs.plate_text, 'plate_confidence': obs.plate_confidence, 'ocr_success': obs.ocr_success} for obs in observations]}
    else:
        import datetime
        group_data = {'id': id, 'label': f'Demo Group #{id}', 'type': 'plate', 'plate_text': 'B 1234 XYZ', 'vehicle_type': 'car', 'vehicle_color': 'white', 'verified': False, 'first_seen': datetime.datetime.now() - datetime.timedelta(days=7), 'last_seen': datetime.datetime.now(), 'photos': [{'id': 1, 'image_path': None, 'annotated_path': None, 'plate_image': None, 'driver_image': None, 'timestamp': datetime.datetime.now() - datetime.timedelta(hours=i), 'source_type': ['image', 'webcam', 'video', 'ipcam'][i % 4], 'plate_text': 'B 1234 XYZ', 'plate_confidence': 0.85 - i * 0.05, 'ocr_success': True} for i in range(5)]}
    return render_template('admin/gallery_detail.html', group=group_data)