import os
from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(BASE_DIR, '.env')
print(f'Memeriksa .env di: {env_path}')
print(f'File ditemukan: {os.path.exists(env_path)}')
load_dotenv(env_path, override=True)
print('Kunci ditemukan di environment:')
found_token = False
for key in os.environ:
    if 'PLATE' in key or 'TOKEN' in key:
        print(f" - {key}: {('[DISENSOR]' if os.environ[key] else '[KOSONG]')}")
        if key == 'PLATE_RECOGNIZER_TOKEN':
            found_token = True
if not found_token:
    print('KRITIS: PLATE_RECOGNIZER_TOKEN tidak ditemukan di environment keys.')
else:
    print('SUKSES: PLATE_RECOGNIZER_TOKEN ditemukan.')