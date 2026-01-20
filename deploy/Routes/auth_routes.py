from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from Libs.auth import check_credentials
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('admin.dashboard'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'
        if check_credentials(username, password):
            session['user'] = username
            session.permanent = remember
            flash(f'Selamat datang, {username}!', 'success')
            next_url = session.pop('next', None)
            if next_url:
                return redirect(next_url)
            return redirect(url_for('admin.dashboard'))
        else:
            error = 'Username atau password salah'
            flash(error, 'danger')
    return render_template('auth/login.html', error=error)

@auth_bp.route('/logout')
def logout():
    username = session.pop('user', None)
    session.clear()
    if username:
        flash(f'Sampai jumpa, {username}!', 'info')
    return redirect(url_for('auth.login'))