from flask import Blueprint, request, jsonify
from Models import db, VehicleIdentity, VehicleObservation, AuditLog
from Libs.identity_manager import get_identity_manager
from Libs.config import VEHICLE_UI_PER_PAGE, OBSERVATIONS_PER_PAGE
vehicle_api = Blueprint('vehicle_api', __name__, url_prefix='/api')

@vehicle_api.route('/identities', methods=['GET'])
def get_identities():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', VEHICLE_UI_PER_PAGE, type=int)
    search = request.args.get('search', '').strip()
    status = request.args.get('status', 'all')
    method = request.args.get('method', 'all')
    query = VehicleIdentity.query
    if search:
        query = query.filter(VehicleIdentity.plate_text.ilike(f'%{search}%'))
    if status == 'verified':
        query = query.filter_by(verified=True)
    elif status == 'unverified':
        query = query.filter_by(verified=False)
    if method in ['plate', 'face', 'visual']:
        query = query.filter_by(identity_method=method)
    query = query.order_by(VehicleIdentity.last_seen.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    return jsonify({'success': True, 'data': [v.to_dict() for v in pagination.items], 'pagination': {'page': pagination.page, 'pages': pagination.pages, 'total': pagination.total, 'per_page': per_page, 'has_next': pagination.has_next, 'has_prev': pagination.has_prev}})

@vehicle_api.route('/identities/<int:id>', methods=['GET'])
def get_identity(id):
    identity = VehicleIdentity.query.get(id)
    if not identity:
        return (jsonify({'success': False, 'error': 'Identitas tidak ditemukan'}), 404)
    include_obs = request.args.get('include_observations', 'true').lower() == 'true'
    return jsonify({'success': True, 'data': identity.to_dict(include_observations=include_obs)})

@vehicle_api.route('/identities/<int:id>/verify', methods=['POST'])
def verify_identity(id):
    identity = VehicleIdentity.query.get(id)
    if not identity and id <= 6:
        return jsonify({'success': True, 'message': 'Identitas berhasil diverifikasi (Dummy)', 'id': id, 'verified': True})
    if not identity:
        return (jsonify({'success': False, 'error': 'Identitas tidak ditemukan'}), 404)
    manager = get_identity_manager()
    result = manager.verify_identity(id, performed_by='api')
    if result['success']:
        return jsonify(result)
    return (jsonify(result), 400)

@vehicle_api.route('/identities/<int:id>/unverify', methods=['POST'])
def unverify_identity(id):
    identity = VehicleIdentity.query.get(id)
    if not identity and id <= 6:
        return jsonify({'success': True, 'message': 'Identitas berhasil dihapus verifikasinya (Dummy)', 'id': id, 'verified': False})
    if not identity:
        return (jsonify({'success': False, 'error': 'Identitas tidak ditemukan'}), 404)
    manager = get_identity_manager()
    result = manager.unverify_identity(id, performed_by='api')
    if result['success']:
        return jsonify(result)
    return (jsonify(result), 400)

@vehicle_api.route('/identities/<int:id>/plate', methods=['PUT'])
def update_plate(id):
    data = request.get_json() or {}
    new_plate = data.get('plate_text', '')
    identity = VehicleIdentity.query.get(id)
    if not identity and id <= 6:
        return jsonify({'success': True, 'message': 'Teks plat berhasil diperbarui (Dummy)', 'id': id, 'plate_text': new_plate})
    if not identity:
        return (jsonify({'success': False, 'error': 'Identitas tidak ditemukan'}), 404)
    manager = get_identity_manager()
    result = manager.update_plate_text(id, new_plate, performed_by='api')
    if result['success']:
        return jsonify(result)
    return (jsonify(result), 400)

@vehicle_api.route('/identities/<int:id>', methods=['DELETE'])
def delete_identity(id):
    identity = VehicleIdentity.query.get(id)
    if not identity and id <= 6:
        return jsonify({'success': True, 'message': 'Identitas berhasil dihapus (Dummy)', 'id': id})
    if not identity:
        return (jsonify({'success': False, 'error': 'Identitas tidak ditemukan'}), 404)
    manager = get_identity_manager()
    result = manager.delete_identity(id, performed_by='api')
    if result['success']:
        return jsonify(result)
    return (jsonify(result), 400)

@vehicle_api.route('/identities/bulk_delete', methods=['POST'])
def bulk_delete_identities():
    data = request.get_json() or {}
    ids = data.get('ids', [])
    if not ids:
        return (jsonify({'success': False, 'error': 'Tidak ada ID yang diberikan'}), 400)
    manager = get_identity_manager()
    success_count = 0
    errors = []
    for id in ids:
        identity = VehicleIdentity.query.get(id)
        if not identity:
            continue
        result = manager.delete_identity(id, performed_by='api')
        if result['success']:
            success_count += 1
        else:
            errors.append(f"Gagal menghapus {id}: {result.get('error')}")
    return jsonify({'success': True, 'deleted_count': success_count, 'errors': errors, 'message': f'{success_count} item berhasil dihapus.'})

@vehicle_api.route('/identities/<int:id>/details', methods=['POST'])
def update_identity_details(id):
    data = request.get_json() or {}
    identity = VehicleIdentity.query.get(id)
    if not identity and id <= 6:
        return jsonify({'success': True, 'message': 'Identitas berhasil diperbarui (Dummy)', 'id': id, 'changes': data})
    if not identity:
        return (jsonify({'success': False, 'error': 'Identitas tidak ditemukan'}), 404)
    manager = get_identity_manager()
    result = manager.update_identity_details(id, data, performed_by='api')
    if result['success']:
        try:
            from Libs.dataset_manager import get_dataset_manager
            ds_manager = get_dataset_manager()
            csv_data = {'plate_number': identity.plate_text, 'vehicle_type': data.get('vehicle_type', identity.vehicle_type), 'vehicle_color': data.get('vehicle_color', identity.vehicle_color), 'driver_name': data.get('driver_name', identity.driver_name), 'driver_id': data.get('driver_id_card', identity.driver_id_card), 'face_status': data.get('driver_face_status', identity.driver_face_status), 'vehicle_box': data.get('vehicle_box', []), 'plate_box': data.get('plate_box', []), 'face_box': data.get('face_box', []), 'body_box': data.get('body_box', [])}
            ds_manager.save_verified_data(csv_data)
            best_obs = identity.observations.filter(VehicleObservation.image_path != None).order_by(VehicleObservation.timestamp.desc()).first()
            if best_obs:
                import json as json_lib

                def get_existing_box(attr_name, obs):
                    val = getattr(identity, attr_name)
                    if val:
                        return json_lib.loads(val)
                    if obs:
                        obs_val = getattr(obs, attr_name)
                        if obs_val:
                            return json_lib.loads(obs_val)
                    return []
                existing_vehicle_box = get_existing_box('vehicle_box', best_obs)
                existing_plate_box = get_existing_box('plate_box', best_obs)
                existing_face_box = get_existing_box('face_box', best_obs)
                existing_body_box = get_existing_box('body_box', best_obs)
                vehicle_box_new = data.get('vehicle_box')
                plate_box_new = data.get('plate_box')
                face_box_new = data.get('face_box')
                body_box_new = data.get('body_box')
                vehicle_box = vehicle_box_new if vehicle_box_new and len(vehicle_box_new) > 0 else existing_vehicle_box
                plate_box = plate_box_new if plate_box_new and len(plate_box_new) > 0 else existing_plate_box
                face_box = face_box_new if face_box_new and len(face_box_new) > 0 else existing_face_box
                body_box = body_box_new if body_box_new and len(body_box_new) > 0 else existing_body_box
                burned_path = ds_manager.generate_burned_annotation(best_obs, vehicle_box=vehicle_box, plate_box=plate_box, face_box=face_box, body_box=body_box)
                if burned_path:
                    best_obs.annotated_image_path = burned_path
                    if vehicle_box:
                        vehicle_crop_path = ds_manager.generate_vehicle_crop(best_obs, vehicle_box)
                        if vehicle_crop_path:
                            best_obs.image_path = vehicle_crop_path
                    try:
                        if face_box and len(face_box) == 4:
                            face_crop_path = ds_manager.generate_face_crop(best_obs, face_box)
                            if face_crop_path:
                                best_obs.driver_image_path = face_crop_path
                        if body_box and len(body_box) == 4:
                            body_crop_path = ds_manager.generate_body_crop(best_obs, body_box)
                            if body_crop_path:
                                if not best_obs.driver_image_path:
                                    best_obs.driver_image_path = body_crop_path
                                else:
                                    best_obs.plate_image_path = body_crop_path
                    except Exception:
                        pass
                    if face_box:
                        best_obs.face_detected = True
                    if face_box or body_box:
                        best_obs.driver_detected = True
                    import json as json_lib
                    if vehicle_box:
                        best_obs.vehicle_box = json_lib.dumps(vehicle_box)
                    if plate_box:
                        best_obs.plate_box = json_lib.dumps(plate_box)
                    if face_box:
                        best_obs.face_box = json_lib.dumps(face_box)
                    if body_box:
                        best_obs.body_box = json_lib.dumps(body_box)
                    best_obs.is_manually_annotated = True
                import json
                if vehicle_box:
                    identity.vehicle_box = json.dumps(vehicle_box)
                if plate_box:
                    identity.plate_box = json.dumps(plate_box)
                if face_box:
                    identity.face_box = json.dumps(face_box)
                    identity.driver_face_status = 'verified'
                if body_box:
                    identity.body_box = json.dumps(body_box)
                identity.is_manually_annotated = True
                db.session.commit()
        except Exception as e:
            print(f'[API] Error saat menyimpan dataset terverifikasi: {e}')
    if result['success']:
        response_data = result.copy()
        clean_url = None
        if identity.observations.count() > 0:
            best_obs = identity.observations.filter(VehicleObservation.image_path != None).order_by(VehicleObservation.timestamp.desc()).first()
            if best_obs:
                src_rel = best_obs.image_path or best_obs.frame_image_path
                if src_rel:
                    from flask import url_for
                    if str(src_rel).lstrip('/').startswith('results/'):
                        clean_url = '/' + str(src_rel).lstrip('/')
                    elif 'static/' in str(src_rel):
                        filename = str(src_rel).split('static/')[-1]
                        clean_url = url_for('static', filename=filename)
                    else:
                        clean_url = '/' + str(src_rel).lstrip('/')
        try:
            import json as _json
        except Exception:
            _json = None

        def _safe_load(s):
            if not s:
                return []
            try:
                return _json.loads(s) if _json else s
            except Exception:
                return s
        response_data.update({'vehicle': identity.to_dict(), 'image_url': clean_url, 'annotations': {'vehicle_box': vehicle_box if 'vehicle_box' in locals() and vehicle_box else _safe_load(identity.vehicle_box), 'plate_box': plate_box if 'plate_box' in locals() and plate_box else _safe_load(identity.plate_box), 'face_box': face_box if 'face_box' in locals() and face_box else _safe_load(identity.face_box), 'body_box': body_box if 'body_box' in locals() and body_box else _safe_load(identity.body_box)}, 'updated_data': {'plate_text': data.get('plate_text', identity.plate_text), 'vehicle_type': data.get('vehicle_type', identity.vehicle_type), 'vehicle_color': data.get('vehicle_color', identity.vehicle_color), 'driver_name': data.get('driver_name', identity.driver_name), 'driver_id_card': data.get('driver_id_card', identity.driver_id_card), 'driver_face_status': data.get('driver_face_status', identity.driver_face_status)}, 'is_verified': True})
        return jsonify(response_data)
    return (jsonify(result), 400)

@vehicle_api.route('/identities/merge', methods=['POST'])
def merge_identities():
    data = request.get_json() or {}
    primary_id = data.get('primary_id')
    secondary_ids = data.get('secondary_ids', [])
    if not primary_id:
        return (jsonify({'success': False, 'error': 'primary_id diperlukan'}), 400)
    if not secondary_ids:
        return (jsonify({'success': False, 'error': 'secondary_ids diperlukan'}), 400)
    manager = get_identity_manager()
    result = manager.merge_identities(primary_id, secondary_ids, performed_by='api')
    if result['success']:
        return jsonify(result)
    return (jsonify(result), 400)

@vehicle_api.route('/identities/split', methods=['POST'])
def split_identity():
    data = request.get_json() or {}
    identity_id = data.get('identity_id')
    observation_ids = data.get('observation_ids', [])
    if not identity_id:
        return (jsonify({'success': False, 'error': 'identity_id diperlukan'}), 400)
    if not observation_ids:
        return (jsonify({'success': False, 'error': 'observation_ids diperlukan'}), 400)
    manager = get_identity_manager()
    result = manager.split_identity(identity_id, observation_ids, performed_by='api')
    if result['success']:
        return jsonify(result)
    return (jsonify(result), 400)

@vehicle_api.route('/observations', methods=['GET'])
def get_observations():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', OBSERVATIONS_PER_PAGE, type=int)
    identity_id = request.args.get('identity_id', type=int)
    source = request.args.get('source', 'all')
    ocr = request.args.get('ocr', 'all')
    query = VehicleObservation.query
    if identity_id:
        query = query.filter_by(vehicle_id=identity_id)
    if source in ['image', 'video', 'webcam', 'ipcam']:
        query = query.filter_by(source_type=source)
    if ocr == 'success':
        query = query.filter_by(ocr_success=True)
    elif ocr == 'failed':
        query = query.filter_by(ocr_success=False)
    query = query.order_by(VehicleObservation.timestamp.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    return jsonify({'success': True, 'data': [o.to_dict() for o in pagination.items], 'pagination': {'page': pagination.page, 'pages': pagination.pages, 'total': pagination.total, 'per_page': per_page, 'has_next': pagination.has_next, 'has_prev': pagination.has_prev}})

@vehicle_api.route('/observations/<int:id>', methods=['GET'])
def get_observation(id):
    observation = VehicleObservation.query.get(id)
    if not observation:
        return (jsonify({'success': False, 'error': 'Observasi tidak ditemukan'}), 404)
    return jsonify({'success': True, 'data': observation.to_dict()})

@vehicle_api.route('/observations/<int:id>', methods=['DELETE'])
def delete_observation(id):
    observation = VehicleObservation.query.get(id)
    if not observation:
        return (jsonify({'success': False, 'error': 'Observasi tidak ditemukan'}), 404)
    identity_id = observation.vehicle_id
    identity = observation.identity
    if identity:
        identity.detection_count = max(0, identity.detection_count - 1)
    audit = AuditLog(action='delete', entity_type='observation', entity_id=id, performed_by='api')
    db.session.add(audit)
    db.session.delete(observation)
    db.session.commit()
    return jsonify({'success': True, 'deleted_observation_id': id, 'identity_id': identity_id})

@vehicle_api.route('/stats', methods=['GET'])
def get_stats():
    from sqlalchemy import func
    total_identities = VehicleIdentity.query.count()
    verified_count = VehicleIdentity.query.filter_by(verified=True).count()
    total_observations = VehicleObservation.query.count()
    method_stats = db.session.query(VehicleIdentity.identity_method, func.count(VehicleIdentity.id)).group_by(VehicleIdentity.identity_method).all()
    type_stats = db.session.query(VehicleIdentity.vehicle_type, func.count(VehicleIdentity.id)).group_by(VehicleIdentity.vehicle_type).all()
    ocr_attempted = VehicleObservation.query.filter_by(ocr_attempted=True).count()
    ocr_success = VehicleObservation.query.filter_by(ocr_success=True).count()
    return jsonify({'success': True, 'data': {'total_identities': total_identities, 'verified_identities': verified_count, 'unverified_identities': total_identities - verified_count, 'total_observations': total_observations, 'identity_methods': {m: c for (m, c) in method_stats}, 'vehicle_types': {t or 'unknown': c for (t, c) in type_stats}, 'ocr_stats': {'attempted': ocr_attempted, 'success': ocr_success, 'success_rate': round(ocr_success / ocr_attempted * 100, 2) if ocr_attempted > 0 else 0}}})

@vehicle_api.route('/audit', methods=['GET'])
def get_audit_log():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    action = request.args.get('action', '')
    query = AuditLog.query
    if action:
        query = query.filter_by(action=action)
    query = query.order_by(AuditLog.timestamp.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    return jsonify({'success': True, 'data': [a.to_dict() for a in pagination.items], 'pagination': {'page': pagination.page, 'pages': pagination.pages, 'total': pagination.total}})

@vehicle_api.route('/observations/<int:obs_id>/move', methods=['POST'])
def move_observation_to_identity(obs_id):
    data = request.get_json() or {}
    new_identity_id = data.get('identity_id')
    if not new_identity_id:
        return (jsonify({'success': False, 'error': 'identity_id diperlukan'}), 400)
    observation = VehicleObservation.query.get(obs_id)
    if not observation:
        return (jsonify({'success': False, 'error': 'Observasi tidak ditemukan'}), 404)
    new_identity = VehicleIdentity.query.get(new_identity_id)
    if not new_identity:
        return (jsonify({'success': False, 'error': 'Identitas tidak ditemukan'}), 404)
    old_identity_id = observation.vehicle_id
    observation.vehicle_id = new_identity_id
    observation.plate_text = new_identity.plate_text
    observation.vehicle_type = new_identity.vehicle_type
    observation.vehicle_color = new_identity.vehicle_color
    if old_identity_id:
        old_identity = VehicleIdentity.query.get(old_identity_id)
        if old_identity:
            old_identity.detection_count = max(0, old_identity.detection_count - 1)
            remaining_obs = old_identity.observations.order_by(VehicleObservation.timestamp.desc()).first()
            if remaining_obs:
                old_identity.last_seen = remaining_obs.timestamp
    new_identity.detection_count += 1
    new_identity.last_seen = observation.timestamp
    if not new_identity.first_seen or observation.timestamp < new_identity.first_seen:
        new_identity.first_seen = observation.timestamp
    import json
    audit = AuditLog(action='move_observation', entity_type='observation', entity_id=obs_id, details=json.dumps({'from_identity': old_identity_id, 'to_identity': new_identity_id, 'plate_text': new_identity.plate_text}), performed_by='admin')
    db.session.add(audit)
    db.session.commit()
    try:
        from Libs.dataset_manager import get_dataset_manager
        dm = get_dataset_manager()
        dm.save_verified_data({'plate_number': new_identity.plate_text or '', 'vehicle_type': new_identity.vehicle_type or '', 'vehicle_color': new_identity.vehicle_color or '', 'driver_name': new_identity.driver_name or '', 'driver_id': new_identity.driver_id_card or '', 'face_status': new_identity.driver_face_status or 'not_available'})
    except Exception as e:
        print(f'Error saat sinkronisasi ke CSV: {e}')
    return jsonify({'success': True, 'observation_id': obs_id, 'new_identity_id': new_identity_id, 'old_identity_id': old_identity_id, 'message': 'Observasi berhasil dipindahkan'})

@vehicle_api.route('/identities/create', methods=['POST'])
def create_new_identity():
    data = request.get_json() or {}
    plate_text = data.get('plate_text', '').strip()
    driver_name = data.get('driver_name', '').strip()
    vehicle_type = data.get('vehicle_type', 'car')
    vehicle_color = data.get('vehicle_color', 'unknown')
    if not plate_text and (not driver_name):
        return (jsonify({'success': False, 'error': 'Nomor plat atau nama pengemudi diperlukan'}), 400)
    if plate_text:
        existing = VehicleIdentity.query.filter_by(plate_text=plate_text).first()
        if existing:
            return (jsonify({'success': False, 'error': f'Identitas dengan plat {plate_text} sudah ada'}), 400)
    from datetime import datetime
    identity = VehicleIdentity(plate_text=plate_text if plate_text else None, vehicle_type=vehicle_type, vehicle_color=vehicle_color, driver_name=driver_name if driver_name else None, identity_method='plate' if plate_text else 'visual', detection_count=0, verified=False, first_seen=datetime.utcnow(), last_seen=datetime.utcnow())
    db.session.add(identity)
    db.session.commit()
    import json
    audit = AuditLog(action='create_identity', entity_type='identity', entity_id=identity.id, details=json.dumps({'plate_text': plate_text, 'driver_name': driver_name, 'vehicle_type': vehicle_type, 'vehicle_color': vehicle_color}), performed_by='admin')
    db.session.add(audit)
    db.session.commit()
    return jsonify({'success': True, 'identity': identity.to_dict(), 'message': 'Grup identitas baru berhasil dibuat'})

@vehicle_api.route('/vehicles/export', methods=['GET'])
def export_vehicles():
    import io
    import csv
    from flask import Response
    from sqlalchemy import func
    try:
        vehicles_query = VehicleIdentity.query.outerjoin(VehicleObservation).group_by(VehicleIdentity.id).add_columns(func.avg(VehicleObservation.plate_confidence).label('avg_confidence'), func.count(VehicleObservation.id).label('observation_count')).all()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Plat Nomor', 'Jenis', 'Warna', 'Pengemudi', 'Keyakinan', 'Status', 'Jumlah Observasi'])
        for item in vehicles_query:
            vehicle = item[0]
            avg_conf = item.avg_confidence
            obs_count = item.observation_count
            if vehicle.is_manually_annotated:
                confidence_str = 'Verified'
            elif avg_conf:
                confidence_str = f'{avg_conf * 100:.0f}%'
            else:
                confidence_str = '-'
            writer.writerow([vehicle.id, vehicle.plate_text or '-', vehicle.vehicle_type or '-', vehicle.vehicle_color or '-', vehicle.driver_name or '-', confidence_str, 'Verified' if vehicle.verified else 'Auto', obs_count or 0])
        output.seek(0)
        return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=kendaraan_export.csv'})
    except Exception as e:
        return (jsonify({'success': False, 'error': str(e)}), 500)

@vehicle_api.route('/reset_data', methods=['DELETE'])
def reset_data():
    try:
        num_obs = db.session.query(VehicleObservation).delete()
        num_identities = db.session.query(VehicleIdentity).delete()
        num_logs = db.session.query(AuditLog).delete()
        db.session.commit()
        audit = AuditLog(action='reset_data', entity_type='system', entity_id=0, details='All data cleared by admin', performed_by='admin')
        db.session.add(audit)
        db.session.commit()
        return jsonify({'success': True, 'message': f'Data berhasil dihapus. {num_obs} observasi, {num_identities} identitas, {num_logs} log audit dihapus.', 'details': {'observations_deleted': num_obs, 'identities_deleted': num_identities, 'audit_logs_deleted': num_logs}})
    except Exception as e:
        db.session.rollback()
        return (jsonify({'success': False, 'error': str(e)}), 500)