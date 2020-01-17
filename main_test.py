# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.30
# set_session(tf.Session(config=config))

import torch
torch.cuda.empty_cache()
# from Rec_text import rec_txt
# from crnn_torch.model import predict
# from ctpn.ctpn_blstm_test import text_predict
# from densent_ocr.model import predict
from direction_detection.direction_correct import fourier_demo
# from pan.predict import text_predict
# from crnn_torch.model import predict

from select_infomodel import select
from warp_templates.get_text import get_texts

from wce_data import WCE
from viterbi import calculate
# from red_method import seal_eliminate

import numpy as np
from PIL import Image
import fitz
import cv2
import requests
import base64
from full_ocr_local import single_ocr
from template_warp import get_warp
import json
import pickle
from sanic import Sanic, response
from config_url import DETECT_URL, RECOGNISE_URL


# a = Image.open('test.png').convert('RGB')
# ori_w, ori_h = a.size
# b = a.resize((1000, 1000))
# scale_w, scale_h = b.size
# b = np.array(b)
# a = np.array(a)
# scale_w, scale_h = ori_w / scale_w, ori_h / scale_h
# text_predict(b, scale_w, scale_h, a)


app = Sanic(__name__)

# with open('info.pkl', 'rb') as pkl:
#     info = pickle.load(pkl)
# info = pickle.load(open('info.pkl', 'rb'))

large_FT = ['FT001001001002', 'FT001001001003', 'FT001001001005', 'FT001003002001', 'FT001001002001', 'FT001001003001']


# @app.route('/fre', methods=['POST'])
# def get_text(request):
#     img_path = request.form.get('input')
#     print(request.form.get('input'))
#     par = request.form.get('par')
#     page = request.form.get('page')
#     print(img_path)
#     try:
#         if img_path.lower().endswith('.pdf'):
#             pdf = fitz.open(img_path)
#             page_num = pdf[int(page) - 1]
#             trans = fitz.Matrix(3, 3).preRotate(0)
#             pm = page_num.getPixmap(matrix=trans, alpha=False)
#             input_img = fourier_demo(Image.frombytes("RGB", [pm.width, pm.height], pm.samples), 'FT001')
#         else:
#             input_img = fourier_demo(Image.open(img_path).convert('RGB'), 'FT001')
#         image = input_img.copy()
#         image = np.array(image)
#         image = corp_margin(image)
#         Image.fromarray(image).save('ttt.jpg')
#         image = Image.fromarray(image).resize((299, 299))
#         FT = predict_all(image, par)
#         return response.json({'result': 'true', 'Images': [{'path': img_path, 'FT': i[0], 'proportion': str(i[1])} for i in FT]})
#     except Exception as e:
#         print(e)
#         return response.json({'result': 'false', 'Images': []})


@app.route('/fce', methods=['POST'])
def get_text(request):
    print('dddddddddddddddddddddddd')
    img_path = request.form.get('img_path')
    par = request.form.get('par')
    task_id = request.form.get('task_id')
    FT = request.form.get('FT')
    page = request.form.get('page')
    print(img_path)
    try:
        if img_path.lower().endswith('.pdf'):
            pdf = fitz.open(img_path)
            page_num = pdf[int(page) - 1]
            trans = fitz.Matrix(3, 3).preRotate(0)
            pm = page_num.getPixmap(matrix=trans, alpha=False)
            ori_img = fourier_demo(Image.frombytes("RGB", [pm.width, pm.height], pm.samples), 'FT001')
        else:
            ori_img = fourier_demo(Image.open(img_path).convert('RGB'), 'FT001')
        ft = select(FT[:11] + '001')
        print('FT:', FT[:11] + '001')

        # input_img = input_img.resize((2000, 2000), Image.ANTIALIAS)
        input_img = ori_img.copy()
        ori_w, ori_h = ori_img.size
        # data_image = str(os.path.splitext(img_path)[0].split('/')[-1]) + '_' + str(page)
        # data_image = '/home/ddwork/wce_data/ori_images/{}_{}.jpg'.format(data_image, task_id)
        # input_img.save(data_image)
        # input_img = np.array(input_img)
        ori_img = np.array(ori_img)
        _, ori = cv2.imencode('.jpg', ori_img)
        ori = base64.b64encode(ori.tostring())
        # input_img = seal_eliminate(input_img)
        import time
        start = time.time()
        while_i = 0
        images = []
        while 1:
            input_img.thumbnail((2000 - while_i * 100, 2000 - while_i * 100), Image.ANTIALIAS)
            # img_name = img_name.resize((1500, 1500), Image.ANTIALIAS)
            # img_name = img_name.convert('RGB')
            scale_w, scale_h = input_img.size
            # print(scale_w, scale_h)
            scale_w, scale_h = ori_w / scale_w, ori_h / scale_h
            print('原图大小:', ori_w, ori_h, '缩放比例:', scale_w, scale_h)
            img = np.array(input_img)
            # B_channel, G_channel, R_channel = cv2.split(img)
            # cv2.imwrite('test.png', R_channel)
            # img = cv2.cvtColor(R_channel, cv2.COLOR_GRAY2BGR)
            _, img = cv2.imencode('.jpg', img)
            img = base64.b64encode(img.tostring())
            data = {'img': img, 'scale_w': scale_w, 'scale_h': scale_h, 'ori_img': ori}
            images_json = requests.post(DETECT_URL, data=data)
            # images = text_predict(img, scale_w, scale_h, ori_img)
            torch.cuda.empty_cache()
            while_i += 1
            if images_json.json() != '':
                for i in images_json.json():
                    image = base64.b64decode(i[1])
                    image = np.fromstring(image, np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    images.append([i[0], image])
                break
        print(111111111111111111111111111, time.time() - start)
        start = time.time()
        # image_positions = [[i[0].tolist(), rec_txt(i[1]).replace('“', '').replace('"', '')] for i
        #                    in
        #                    images]
        image_positions = []
        for index, j in enumerate(images):
            # if j[1].any() and j[1].shape[0] < j[1].shape[1] * 1.5:
            try:

                _, img = cv2.imencode('.jpg', j[1])
                img = base64.b64encode(img.tostring())
                data = {'img': img}
                content = requests.post(RECOGNISE_URL, data=data).json()[:2]
                # ori_content = [i[0] for i in content[0]]
                # prob_content = [[i, j] for i, j in zip(content[0], content[1])]
                for indexi, i in enumerate(content[1]):
                    if i[0] > 0.9:
                        content[0][indexi] = content[0][indexi][0]
                        content[1][indexi] = [-1]
                while 1:
                    try:
                        content[1].remove([-1])
                    except:
                        break
                content = calculate(content)

                image_positions.append([j[0], content.replace('“', '').replace('‘', '')])
            except Exception as e:
                print('h w', e)
                continue
        # torch.cuda.empty_cache()
        # data_json[task_id] = [par, data_image, FT, image_positions]
        # data_json = WCE.create(field_id=int(task_id), par=str(par), image_path=data_image, FT=FT, file_type=FT[:11],
        #                        image_positions=str(image_positions), edited=False, trained=False)
        # data_json.save()
        print(222222222222222222222222222, time.time() - start)
        text = single_ocr(image_positions)
        print(text)
        # with open(img_path + '.txt', 'w', encoding='utf-8') as fd:
        #     fd.write(text)
        texts = ft.extract_info(img_path, page, FT[:11] + '001', text)
        print(texts)
        # try:
        #     found = get_warp(input_img, image_positions, FT)
        #     print('ssssssssssssssssssssssss')
        #     found_texts = get_texts('warp_templates/{}/template.xml'.format(FT), found, img_path, task_id)
        # except Exception as e:
        found_texts = ''
        # print(e)
        print('==================================================================')
        print(texts, found_texts)
        torch.cuda.empty_cache()
        # 资质证书判断
        if FT[:11] == 'FT001003110':
            FT = FT[:8] + texts.get('version')
        # 路径中取日期
        try:
            if texts.get('发证日期') == '' or not texts.get('发证日期'):
                import re
                date_path = re.search('([0-9]{4}[-/年][0-9]{1,2}[-/月][0-9]{1,2}日?)', os.path.split(img_path)[1])
                if date_path:
                    texts['发证日期'] = date_path.groups()[0]
        except:
            pass

        if texts == 'FT999' and found_texts:
            return response.json(
                {'result': 'true', 'message': '请求成功', 'taskid': task_id, 'fields': found_texts, 'FT': FT})
        if texts != 'FT999' and found_texts == '':
            return response.json({'result': 'true', 'message': '请求成功', 'taskid': task_id, 'fields': texts, 'FT': FT})
        if found_texts:
            for key, value in texts.items():
                try:
                    if value == '':
                        texts[key] = found_texts[key]
                except:
                    continue
        blank = 0
        for key, value in texts.items():
            if value == '':
                blank += 1
        if blank == len(texts) - 1:
            return response.json(
                {'result': 'false', 'message': '请求失败', 'taskid': task_id, 'fields': {}, 'FT': 'FT999999999'})
        else:
            return response.json({'result': 'true', 'message': '请求成功', 'taskid': task_id, 'fields': texts, 'FT': FT})
    except Exception as e:
        print(e)
        return response.json(
            {'result': 'false', 'message': '请求失败', 'taskid': task_id, 'fields': {}, 'FT': 'FT999999999'})


@app.route('/check')
def health_check(request):
    return response.text('ok')


if __name__ == '__main__':
    app.config.KEEP_ALIVE = False
    app.config.REQUEST_TIMEOUT = 900
    app.config.RESPONSE_TIMEOUT = 900
    app.run(host='0.0.0.0', port=8005)
