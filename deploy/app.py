import sys
import os
from datetime import timedelta
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from flask import Flask, send_from_directory
app = Flask(__name__, template_folder='Views', static_folder='static')

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename)

@app.template_filter('map_color')
def map_color_filter(color_name):
    if not color_name:
        return '#6c757d'
    c = color_name.lower().strip()
    colors = {'white': '#f8f9fa', 'putih': '#f8f9fa', 'black': '#212529', 'hitam': '#212529', 'silver': '#adb5bd', 'abu-abu': '#adb5bd', 'abu': '#adb5bd', 'red': '#dc3545', 'merah': '#dc3545', 'blue': '#0d6efd', 'biru': '#0d6efd', 'green': '#198754', 'hijau': '#198754', 'yellow': '#ffc107', 'kuning': '#ffc107', 'orange': '#fd7e14', 'jingga': '#fd7e14'}
    return colors.get(c, '#6c757d')

@app.template_filter('text_color')
def text_color_filter(color_name):
    if not color_name:
        return 'white'
    c = color_name.lower().strip()
    if c in ['white', 'putih', 'yellow', 'kuning', 'silver', 'abu-abu', 'abu']:
        return '#000000'
    return '#ffffff'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///vehicle_identity.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.secret_key = 'vehicle-identity-system-secret-key-change-in-production'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
from Models import db
db.init_app(app)
with app.app_context():
    db.create_all()
    print('[APP] Tabel database dibuat/diverifikasi')
from Routes.auth_routes import auth_bp
app.register_blueprint(auth_bp)
from Routes.api import api
app.register_blueprint(api)
from Routes.admin_routes import admin_bp
app.register_blueprint(admin_bp)
from Routes.vehicle_api import vehicle_api
app.register_blueprint(vehicle_api)
from Libs.config import ensure_directories
ensure_directories()
if __name__ == '__main__':
    print(f'[APP] Memulai Sistem Identitas Kendaraan dari {current_dir}')
    print('[APP] Rute yang tersedia:')
    print('  - / (Beranda Tamu)')
    print('  - /login (Halaman Login)')
    print('  - /logout (Logout)')
    print('  - /admin (Dashboard Admin) [Terproteksi]')
    print('  - /admin/vehicles (Identitas Kendaraan) [Terproteksi]')
    print('  - /admin/observations (Semua Observasi) [Terproteksi]')
    print('  - /admin/gallery (Galeri) [Terproteksi]')
    print('  - /admin/merge (Gabung & Pisah) [Terproteksi]')
    print('  - /admin/settings (Pengaturan) [Terproteksi]')
    print('  - /api/... (API JSON)')
    print('[APP] Login default: admin / admin123')
    app.run(host='0.0.0.0', port=5001, debug=True)