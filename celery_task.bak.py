import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.30
set_session(tf.Session(config=config))

import torch
torch.cuda.empty_cache()
# from Rec_text import rec_txt
# from crnn_torch.model import predict
# from ctpn.ctpn_blstm_test import text_predict
# from densent_ocr.model import predict
from direction_detection.direction_correct import fourier_demo
from pan.predict import text_predict
from crnn_torch.model import predict

from select_infomodel import select
# from warp_templates.get_text import get_texts

from wce_data import WCE
from viterbi import calculate
# from red_method import seal_eliminate

import numpy as np
from PIL import Image
import fitz
from full_ocr_local import single_ocr
# from template_warp import get_warp
import json
import pickle
#from sanic import Sanic, response


a = Image.open('test.png').convert('RGB')
ori_w, ori_h = a.size
b = a.resize((1000, 1000))
scale_w, scale_h = b.size
b = np.array(b)
scale_w, scale_h = ori_w / scale_w, ori_h / scale_h
text_predict(b, scale_w, scale_h, a)


#app = Sanic(__name__)

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


#@app.route('/fce', methods=['POST'])
def get_text(request):
    img_path = request.get('img_path')
    par = request.get('par')
    task_id = request.get('task_id')
    FT = request.get('FT')
    page = request.get('page')
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
        f = select(FT[:11] + '001')
        print('FT:', FT[:11] + '001')

        # input_img = input_img.resize((2000, 2000), Image.ANTIALIAS)
        ori_w, ori_h = ori_img.size
        input_img = ori_img.copy()
        input_img.thumbnail((2000, 2000), Image.ANTIALIAS)
        # input_img = input_img.resize((2000, 2000), Image.ANTIALIAS)
        scale_w, scale_h = input_img.size
        scale_w, scale_h = ori_w / scale_w, ori_h / scale_h
        input_img = input_img.convert('RGB')
        data_image = str(os.path.splitext(img_path)[0].split('/')[-1]) + '_' + str(page)
        data_image = '/home/ddwork/wce_data/ori_images/{}_{}.jpg'.format(data_image, task_id)
        input_img.save(data_image)
        input_img = np.array(input_img)
        # input_img = seal_eliminate(input_img)
        import time
        start = time.time()
        print("text_predict zhiqian")
        images = text_predict(input_img, scale_w, scale_h, ori_img)
        print("text_predict zhihou")
        torch.cuda.empty_cache()
        
        print(111111111111111111111111111,"SAD", time.time() - start,'HAPPY')
        start = time.time()
        # image_positions = [[i[0].tolist(), rec_txt(i[1]).replace('“', '').replace('"', '')] for i
        #                    in
        #                    images]
        image_positions = []
        for j in images:
            try:
                print("predict front!!!!!!!!!!!!!!")
                content = predict(Image.fromarray(j[1]).convert('L'))
                print("predict back!!!!!!!!!!!!!!")
                for index, i in enumerate(content[1]):
                    if i[0] > 0.9:
                        content[0][index] = content[0][index][0]
                        content[1].pop(index)
                # if i[0] < 0.9:
                #     img = Image.fromarray(j[1]).convert('L')
                #     width, height = img.size[0], img.size[1]
                #     scale = height * 1.0 / 32
                #     width = int(width / scale)
                #
                #     img = img.resize([width, 32], Image.ANTIALIAS)
                #     img = np.array(img)
                #     new_img = img[:, (content[2][index] - 1) * 8:(content[2][index] + 2) * 8]
                #     word, prob = attention(new_img)
                #     if prob > 0.9:
                #         content[0][index] = word[0]
                #         content[1].pop(index)
                # else:
                #     content[0][index] = content[0][index][0]
                #     content[1].pop(index)
                content = calculate(content)
                image_positions.append([j[0], content.replace('“', '').replace('‘', '')])
            except Exception as e:
                print(e)
                continue
        # torch.cuda.empty_cache()
        # data_json[task_id] = [par, data_image, FT, image_positions]
        data_json = WCE.create(field_id=int(task_id), par=str(par), image_path=data_image, FT=FT, file_type=FT[:11],
                               image_positions=str(image_positions), edited=False, trained=False)
        data_json.save()
        print(222222222222222222222222222, time.time() - start)
        text = single_ocr(image_positions)
        print(text)
        # with open(img_path + '.txt', 'w', encoding='utf-8') as fd:
        #     fd.write(text)
        texts = f.extract_info(img_path, page, FT[:11] + '001', text)
        print(texts)
        # try:
        #     found = get_warp(input_img, image_positions, FT)
        #     found_texts = get_texts('warp_templates/{}/template.xml'.format(FT), found, img_path, task_id)
        # except Exception as e:
        #     print(e)
        found_texts = ''
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
            return {'result': 'true', 'message': '请求成功', 'taskid': task_id, 'fields': found_texts, 'FT': FT}
        if texts != 'FT999' and found_texts == '':
            return {'result': 'true', 'message': '请求成功', 'taskid': task_id, 'fields': texts, 'FT': FT}
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
            return {'result': 'false', 'message': '请求失败', 'taskid': task_id, 'fields': {}, 'FT': 'FT999999999'}
        else:
            return {'result': 'true', 'message': '请求成功', 'taskid': task_id, 'fields': texts, 'FT': FT}
    except Exception as e:
        print(e)
        return {'result': 'false', 'message': '请求失败', 'taskid': task_id, 'fields': {}, 'FT': 'FT999999999'}


#@app.route('/check')
#def health_check(request):
#    return response.text('ok')


#if __name__ == '__main__':
#    app.config.KEEP_ALIVE = False
#    app.config.REQUEST_TIMEOUT = 900
#    app.config.RESPONSE_TIMEOUT = 900
#    app.run(host='0.0.0.0', port=8005)
from celery import Celery
import requests
import json
import os
import time
import socket
import urllib

ddwork = Celery('sh_tasks', broker='redis://172.30.81.208:30443/0', backend='redis://172.30.81.208:30445/1')
ddwork.conf.CELERYD_FORCE_EXECV = True

ddwork.conf.BROKER_TRANSPORT_OPTIONS = {'visibility_timeout': 36000}

requests.adapters.DEFAULT_RETRIES = 3 #重连次数

@ddwork.task
def fce_callback(url, data, task_id, work_type, token, dbname, state, project_id, timeout=36000):
    print ('URL:{},参数:{}'.format(url,data),task_id)
    odoo_data = {'access_token':token,'dbname':dbname}
    odoo_url = 'http://172.30.81.208:32621/restapi/1.0/extension_object/sh.work.task/ML_action_to_fce_callback'
    headers = {'User-Agent': 'User-Agent:Mozilla/5.0'}
    start_time = time.time()
    try:
        res = get_text(data)
        res = json.dumps(res)
        #res = requests.post(url, data=data, headers=headers, timeout=timeout).text
        print ('URL:{},返回值:{},类型:{}'.format(url,res,type(res)),task_id)
        #回调
        if work_type == 'fce':
            odoo_url = 'http://172.30.81.208:32621/restapi/1.0/extension_object/sh.work.task/ML_action_to_fce_callback'
            odoo_data.update({'result':res,'task_id':task_id,'all_time':str(time.time()-start_time),'state':state})
        odoo_res = requests.post(odoo_url, data=odoo_data, headers=headers, timeout=timeout).text
        print('回调返回值：{}'.format(json.loads(odoo_res)))
        return odoo_data
    except Exception as e:
        if not os.path.exists(r'/home/ddwork/projects/compound_log'):
            os.makedirs('/home/ddwork/projects/compound_log')
        if not os.path.exists(r'/home/ddwork/projects/compound_log/{}'.format(project_id)):
            os.makedirs('/home/ddwork/projects/compound_log/{}'.format(project_id))
        if not os.path.exists(r'/home/ddwork/projects/compound_log/{}/FCE'.format(project_id)):
            os.makedirs('/home/ddwork/projects/compound_log/{}/FCE'.format(project_id))
        with open(r'/home/ddwork/projects/compound_log/{}/FCE/{}_log.txt'.format(project_id,str(time.strftime('%Y-%m-%d'))), 'a', encoding='utf-8') as gap:
            gap.write(str(time.strftime('%Y-%m-%d %H:%M:%S'))+'\n'+ str(url) +'\n'+ str(data) + '\n'+str(task_id) +'\n'+str(e)+'\n\n')
            if odoo_url:
                gap.write(str(time.strftime('%Y-%m-%d %H:%M:%S'))+'\n'+ str(odoo_url) +'\n'+ str(odoo_data) + '\n'+str(task_id) +'\n'+str(e)+'\n\n')
            gap.close() 
        odoo_data.update({'result':{"res":str(e)},'task_id':task_id,'all_time':str(time.time()-start_time),'state':state})
        odoo_res = requests.post(odoo_url, data=odoo_data, headers=headers, timeout=timeout).text
        print('回调异常返回值：{}'.format(json.loads(odoo_res)))    
        return {'result':str(e),'task_id':task_id,'start_time':start_time}

