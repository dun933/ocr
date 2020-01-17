import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
# set_session(tf.Session(config=config))

from img_cls.predict_2 import predict_all, corp_margin
from direction_detection.direction_correct import fourier_demo

import numpy as np
from PIL import Image
import fitz
from sanic import Sanic, response

app = Sanic(__name__)


@app.route('/fre', methods=['POST'])
def get_text(request):
    img_path = request.form.get('input')
    print(request.form.get('input'))
    par = request.form.get('par')
    page = request.form.get('page')
    print(img_path)
    try:
        if img_path.lower().endswith('.pdf'):
            pdf = fitz.open(img_path)
            page_num = pdf[int(page) - 1]
            trans = fitz.Matrix(3, 3).preRotate(0)
            pm = page_num.getPixmap(matrix=trans, alpha=False)
            input_img = fourier_demo(Image.frombytes("RGB", [pm.width, pm.height], pm.samples), 'FT001')
        else:
            input_img = fourier_demo(Image.open(img_path).convert('RGB'), 'FT001')
        image = input_img.copy()
        image = np.array(image)
        image = corp_margin(image)
        Image.fromarray(image).save('ttt.jpg')
        image = Image.fromarray(image).resize((299, 299))
        FT = predict_all(image, par)
        return response.json({'result': 'true', 'Images': [{'path': img_path, 'FT': i[0], 'proportion': str(i[1])} for i in FT]})
    except Exception as e:
        print(e)
        return response.json({'result': 'false', 'Images': []})


if __name__ == '__main__':
    app.config.KEEP_ALIVE = False
    app.config.REQUEST_TIMEOUT = 900
    app.config.RESPONSE_TIMEOUT = 900
    app.run(host='0.0.0.0', port=8008)
