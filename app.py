# app.py
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from knn_model import klasifikasi_dan_segmentasi

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        original_image=None,
        segmented_image=None,
        mask_image=None,
        fitur_rgb=None,
        predicted_label=None
    )


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(url_for('upload'))

        file = request.files['image']
        if file.filename == '':
            return redirect(url_for('upload'))

        if not (file and allowed_file(file.filename)):
            return redirect(url_for('upload'))

        filename = secure_filename(file.filename)
        original_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(original_path)

        pil_image = Image.open(original_path)

        img_np, img_seg_np, mask_np, fitur_rgb, pred_label = \
            klasifikasi_dan_segmentasi(pil_image)

        seg_img = Image.fromarray(img_seg_np)
        seg_filename = f"seg_{filename}"
        seg_path = os.path.join(RESULT_FOLDER, seg_filename)
        seg_img.save(seg_path)

        mask_img_pil = Image.fromarray(mask_np)
        mask_filename = f"mask_{filename}"
        mask_path = os.path.join(RESULT_FOLDER, mask_filename)
        mask_img_pil.save(mask_path)

        return render_template(
            'upload.html',
            original_image=filename,
            segmented_image=seg_filename,
            mask_image=mask_filename,
            fitur_rgb=fitur_rgb.tolist(),
            predicted_label=pred_label
        )
    else:  # GET request
        return render_template('upload.html')




if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True, use_reloader=False)
