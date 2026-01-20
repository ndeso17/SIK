from functools import wraps
from flask import session, redirect, url_for, request, flash
ADMIN_USERS = {'admin': 'admin123', 'operator': 'operator123'}

def login_required(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Silakan login terlebih dahulu', 'warning')
            session['next'] = request.url
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

def check_credentials(username, password):
    if username in ADMIN_USERS:
        return ADMIN_USERS[username] == password
    return False

def get_current_user():
    return session.get('user')

def is_logged_in():
    return 'user' in session