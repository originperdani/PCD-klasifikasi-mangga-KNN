# app.py
import os
import base64
from io import BytesIO
import tempfile
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from knn_model import klasifikasi_dan_segmentasi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, 'static'),
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_url_path='/static'
)

TMP_DIR = (
    os.environ.get('TMPDIR')
    or os.environ.get('TEMP')
    or os.environ.get('TMP')
    or tempfile.gettempdir()
)
UPLOAD_FOLDER = os.path.join(TMP_DIR, 'uploads')
RESULT_FOLDER = os.path.join(TMP_DIR, 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def to_data_uri(pil_img, fmt='PNG'):
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/{fmt.lower()};base64,{b64}"


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
        seg_data_uri = to_data_uri(seg_img, fmt='PNG')

        mask_img_pil = Image.fromarray(mask_np)
        mask_data_uri = to_data_uri(mask_img_pil, fmt='PNG')

        original_data_uri = to_data_uri(pil_image.convert('RGB'), fmt='PNG')

        return render_template(
            'upload.html',
            original_data_uri=original_data_uri,
            segmented_data_uri=seg_data_uri,
            mask_data_uri=mask_data_uri,
            fitur_rgb=fitur_rgb.tolist(),
            predicted_label=pred_label
        )
    else:  # GET request
        return render_template('upload.html')




if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True, use_reloader=False)
