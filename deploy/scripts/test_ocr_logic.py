import sys
import os
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Libs.pipeline import Pipeline

def test_ocr():
    print('Menginisialisasi Pipeline...')
    pipeline = Pipeline()
    print('\n--- Tes Kasus 1: Plat Indonesia Valid (Tesseract Seharusnya Berhasil) ---')
    img1 = np.ones((64, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img1, 'B 1234 CD', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    (text1, conf1, source1) = pipeline.read_plate_text(img1, debug=True)
    print(f"Hasil 1: '{text1}' (Sumber: {source1})")
    print('\n--- Tes Kasus 2: Suffix Tidak Valid (4 Huruf) (Seharusnya Fallback) ---')
    img2 = np.ones((64, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img2, 'B 1234 CDEF', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    (text2, conf2, source2) = pipeline.read_plate_text(img2, debug=True)
    print(f"Hasil 2: '{text2}' (Sumber: {source2})")
    print('\n--- Tes Kasus 3: Prefix Tidak Valid (3 Huruf) (Seharusnya Fallback) ---')
    img3 = np.ones((64, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img3, 'ABC 1234 CD', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    (text3, conf3, source3) = pipeline.read_plate_text(img3, debug=True)
    print(f"Hasil 3: '{text3}' (Sumber: {source3})")
if __name__ == '__main__':
    test_ocr()