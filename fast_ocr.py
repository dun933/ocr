import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config = tf.ConfigProto(
    log_device_placement=True,
    inter_op_parallelism_threads=0,
    intra_op_parallelism_threads=0,
    allow_soft_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.30
config.gpu_options.allow_growth = False
set_session(tf.Session(config=config))

from math import fabs, sin, radians, cos
import time
from datetime import datetime
import base64
import requests
import cv2
import json
import torch
torch.cuda.empty_cache()
# from pdf2image import convert_from_path
# from Rec_text import rec_txt
# from ctpn.ctpn_blstm_test import text_predict
from pan.predict import text_predict
from crnn_torch.model1 import predict
# from densent_ocr.model import predict
from extract_table_1 import extract_table, generate_table
import numpy as np
from PIL import Image
import fitz
from viterbi import calculate
# from config_url import DETECT_URL, RECOGNISE_URL
# from unet_table.table_line import unet_table
from sanic import Sanic, response

app = Sanic(__name__)


def rotate_img(image, degree):
    degree = -degree
    img1 = np.array(image.convert('RGB'))
    height, width = img1.shape[:2]

    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  # 这个公式参考之前内容
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(img1, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return Image.fromarray(imgRotation)


def layout(texts, im_w, table_infos):
    if len(table_infos) > 0:
        print('table_infos', table_infos)
        for table in table_infos:
            for i in range(len(texts)-1, -1, -1):
                if table[0][0][1] < (texts[i][0][0][1]+texts[i][0][0][3])/2 < table[0][0][3]:
                    del texts[i]

    new_texts = []
    for i in texts:
        end_flg = 'general'
        if re.compile('(^([0-9]{1,2}|[一二三四五六七八九十]{1,3})[、.])|(^[A-Z][）)])').search(i[1]) or (
            re.search(r'(^第.{1,3}条)|(^[(（][0-9一二三四五六七八九十]{1,3}[）)])', i[1])):
            if len(new_texts) > 0:
                new_texts[-1][-1] = 'sub'
        if i[0][0][2] < im_w * 0.7:
            end_flg = 'sub'
        new_texts.append([i[0], i[1], end_flg])

    for table in table_infos:
        if len(new_texts) == 0:
            new_texts.append([table[0], table[1], 'table'])
            continue
        for i in range(len(new_texts)-1, -1, -1):
            if new_texts[i][0][0][1] < table[0][0][1]:
                print(11111111111111)
                new_texts.insert(i+1, [table[0], table[1], 'table'])
                break
            if i == 0:
                new_texts.insert(i, [table[0], table[1], 'table'])

    document = ''
    for i in new_texts:
        end_flg = ''
        if i[-1] == 'sub':
            end_flg = '\n'
        if i[-1] == 'table':
            end_flg = '\n'
            if not document.endswith('\n'):
                document += '\n'
        document += i[1] + end_flg
    return document


def count_area(x, y):
    max_area = abs(x[6] - y[0]) + abs(x[0] - y[6])
    min_area = abs(x[6] - y[6]) + abs(x[0] - y[0])
    area = abs(max_area - min_area) / 2
    return area


# from sanic import Sanic, response
# a = Image.open('test.png').convert('RGB')
# ori_w, ori_h = a.size
# b = a.resize((1000, 1000))
# scale_w, scale_h = b.size
# b = np.array(b)
# scale_w, scale_h = ori_w / scale_w, ori_h / scale_h
# text_predict(b, scale_w, scale_h, a)
# torch.cuda.empty_cache()


@app.route('/fce', methods=['POST'])
def fast_ocr(request):
    ori_time = time.time()
    ori_start_time = datetime.now()
    print('start...')
    img_path = request.form.get('img_path')
    print(img_path)
    print(request.form.get('position'))
    position = '[' + request.form.get('position') + ']'
    rotate = int(request.form.get('rotate'))

    page = request.form.get('pageNum', 1)

    # FT = request.form.get('FT', None)
    # file_type = request.form.get('file_type', None)
    # par_code = request.form.get('par_code', None)
    # project_id = request.form.get('project_id', None)
    #
    # with open('/home/ddwork/projects/compound_log/project_infos/fast_ocr.log', 'a', encoding='utf-8') as f:
    #     f.write(str(FT) + '\t' + str(file_type) + '\t' + str(par_code) + '\t' + str(project_id) + '\t' + str(img_path) + '\t' + str(page) + '\n')
    print(page)
    position = eval(position)
    if img_path.lower().endswith('pdf'):
        image_w = int(request.form.get('imageW'))
        image_h = int(request.form.get('imageH'))
        pdf = fitz.open(img_path)
        page = pdf[int(page) - 1]
        trans = fitz.Matrix(3, 3).preRotate(0)
        pm = page.getPixmap(matrix=trans, alpha=False)
        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
        img = np.array(img)
    else:
        img = np.array(Image.open(img_path).convert('RGB'))
        image_h, image_w = img.shape[:-1]
    print('原图:', img.shape)
    # crop_img = img[position[1]:position[3], position[0]:position[2]]
    # crop_img = np.array(crop_img)
    # crop_img = rotate_img(Image.fromarray(crop_img), rotate).convert('L')
    # ori_img = np.array(crop_img)

    # img = rotate_img(Image.fromarray(img), rotate).convert('L')
    # img = Image.fromarray(img)
    print('rotate', rotate)

    # Image.fromarray(img).save('11111111.jpg')

    img_h, img_w, c = img.shape
    position[0] = position[0] if position[0] > 0 else 0
    position[1] = position[1] if position[0] > 0 else 0
    ori_img = img[int(position[1]*img_h/image_h):int(position[3]*img_h/image_h),
            int(position[0]*img_w/image_w):int(position[2]*img_w/image_w)]
    ori_img = rotate_img(Image.fromarray(ori_img), rotate).convert('L')
    # ori_w, ori_h = ori_img.size
    crop_img = np.array(ori_img.convert('RGB'))
    # Image.fromarray(crop_img).save('11111111.jpg')

    table_infos = []

    # 如果框选高宽比过大，则不考虑表格
    # TODO 判断可能有问题
    # print(111111111111111111, image.shape[0], image.shape[1])
    start_table_time = time.time()
    if crop_img.shape[1] / crop_img.shape[0] > 3 and crop_img.shape[1]/np.array(ori_img).shape[1] < 0.3:
        print('判断不走表格!')
        pass
    else:
        try:
            # 判断是否为表格
            # from config_url import TABLE_URL
            # import base64, requests
            # retval, buffer = cv2.imencode('.jpg', crop_img)
            # pic_str = base64.b64encode(buffer)
            # pic_str = pic_str.decode()
            # r = requests.post(TABLE_URL, data={"img": pic_str})
            # img_byte = base64.b64decode(r.content.decode("utf-8"))
            # img_np_arr = np.fromstring(img_byte, np.uint8)
            # src = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
            tables = extract_table(crop_img)
            texts_table = []
            if tables:
                if tables != 'not table':
                    for table in tables:
                        table_time = time.time()
                        texts_table.append(['table', generate_table(table, crop_img)])
                        print('generate_table time is ', time.time()-table_time)

                for table in texts_table:
                    cell_info = []
                    table_info = [['' for row in range(table[1][2])] for col in range(table[1][1])]
                    for tb in table[1][0]:
                        d = tb[0]
                        for row in range(d['row_begin'], d['row_end']):
                            for col in range(d['col_begin'], d['col_end']):
                                try:
                                    table_info[row][col] += tb[1][0]
                                    if d not in cell_info:
                                        cell_info.append(d)
                                except:
                                    print('cell error')
                    print(f'###start{str(table_info)}end###')
                    x0, y0, x1, y1 = table[-1][-1][0], table[-1][-1][1], table[-1][-1][0]+table[-1][-1][2], \
                                     table[-1][-1][1]+table[-1][-1][3]

                    new_cell_info = []
                    for cell in cell_info:
                        if cell['row_end']-cell['row_begin'] == 1 and cell['col_end']-cell['col_begin'] == 1:
                            continue
                        new_cell_info.append([[cell['row_begin'], cell['col_begin']], [cell['row_end']-1, cell['col_end']-1]])
                    cell_info = new_cell_info

                    table_infos.append([[[x0, y0, x1, y1], [x0, y0, x1, y1]], f'###start{str(table_info)}******{str(cell_info)}end###'])
                    # return response.text(f'###start{str(table_info)}end###')
        except Exception as ex:
            print('table error', ex)
    print('table detect time is ', time.time()-start_table_time)
    # crop_img = cv2.copyMakeBorder(crop_img, int(image_h / 2), int(image_h / 2), int(image_w / 2), int(image_w / 2),
    #                            cv2.BORDER_REPLICATE)
    # short_size = 640
    # h, w = crop_img.shape[:2]
    # short_edge = min(h, w)
    # if short_edge < short_size:
    #     # 保证短边 >= inputsize
    #     scale = short_size / short_edge
    #     if scale > 1:
    #         crop_img = cv2.resize(crop_img, dsize=None, fx=scale, fy=scale)
    # ori_img = np.array(ori_img)
    # _, ori = cv2.imencode('.jpg', ori_img)
    # ori = base64.b64encode(ori.tostring())
    crop_img = Image.fromarray(crop_img)
    while_i = 0
    st_time = time.time()
    # crop_area = []
    while 1:
        crop_img.thumbnail((1500 - while_i * 100, 1500 - while_i * 100), Image.ANTIALIAS)
        # crop_img = crop_img.resize((1500, 1500))
        # scale_w, scale_h = crop_img.size
        # scale_w, scale_h = ori_w / scale_w, ori_h / scale_h
        # crop_img = crop_img.resize((1000, 1000))
        # crop_img.save('111.jpg')
        crop_img = np.array(crop_img)
        # _, img = cv2.imencode('.jpg', crop_img)
        # img = base64.b64encode(img.tostring())
        # data = {'img': img, 'scale_w': 1, 'scale_h': 1, 'ori_img': img}
        # crop_area_json = requests.post(DETECT_URL, data=data)
        # while_i += 1
        # if crop_area_json.json() != '':
        #     for i in crop_area_json.json():
        #         image = base64.b64decode(i[1])
        #         image = np.fromstring(image, np.uint8)
        #         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #         crop_area.append([i[0], image])
        #     break

        crop_area = text_predict(crop_img, 1, 1, crop_img)
        torch.cuda.empty_cache()
        break
    print('ctpn time: ', time.time()-st_time, ' counts: ', len(crop_area))
    # Image.fromarray(crop_img).show()
    # Image.fromarray(crop_area[0][1]).show()
    # save_img = crop_img.copy()
    # for te in crop_area:
    #     # print(1111, te[0])
    #     t = te[0]
    #     cv2.rectangle(save_img, (t[0], t[1]), (t[6], t[7]), (255, 0, 0), 1)
    # Image.fromarray(save_img).save('able2.jpg')
    # # from pan.predict import text_predict
    # img_save = crop_img.copy()
    # sss = text_predict(img_save, 1, 1, img_save)
    # for i in sss:
    #     print(123456, i[0])
    #     cv2.rectangle(img_save, (i[0][0] - 1, i[0][1] - 1), (i[0][6] + 1, i[0][7] + 1), (255, 0, 0), 1)
    # Image.fromarray(img_save).save('able4.jpg')
    new_results = []
    for index, j in enumerate(crop_area):
        # image_positions = [[i[0].tolist(), rec_txt(i[1]).replace('“', '').replace('"', '')] for i
        #                    in
        #                    images]
        try:
            # _, img = cv2.imencode('.jpg', j[1])
            # img = base64.b64encode(img.tostring())
            # data = {'img': img}
            # content = requests.post(RECOGNISE_URL, data=data).json()[:2]
            content, _ = predict(Image.fromarray(j[1]).convert('L'))
            for indexi, i in enumerate(content[1]):
                if i[0] > 0.9:
                    content[0][indexi] = content[0][indexi][0]
                    content[1][indexi] = [-1]
            while 1:
                try:
                    content[1].remove([-1])
                except:
                    break
            print(content)
            content = calculate(content)
            # Image.fromarray(j[1]).save('found/{}.jpg'.format(''.join(img_path.split('/'))))
            # torch.cuda.empty_cache()
            print(content)
            new_results.append([j[0], content])
        except Exception as ex:
            print(ex)
            continue
        # torch.cuda.empty_cache()
        # data_json[task_id] = [par, data_image, FT, image_positions]
    document = ''
    new_results = sorted(new_results, key=lambda i: i[0][1])
    line_images = []
    cut_index = 0
    curr_index = 0
    print(2222222222, len(new_results))
    for index, i in enumerate(new_results):
        try:
            if index == len(new_results) - 1:
                # print(cut_index)
                if cut_index < index:
                    line_images.append(new_results[cut_index:])
                else:
                    line_images.append(new_results[index:])
                break
            # if abs(new_results[index + 1][0][1] - new_results[index][0][1]) > (
            #         new_results[index][0][7] - new_results[index][0][1]) * 4 / 5:
            #     line_images.append(new_results[cut_index: index + 1])
            #     cut_index = index + 1
            if abs(new_results[index + 1][0][1] - new_results[curr_index][0][1]) < (
                    new_results[curr_index][0][7] - new_results[curr_index][0][1]) * 3 / 4:
                for result in new_results[cut_index: index + 1]:
                    if count_area(new_results[index + 1], result) > (result[0][6] - result[0][0])/2:
                        line_images.append(new_results[cut_index: index + 1])
                        cut_index = index + 1
                        curr_index = index + 1
                continue
            else:
                line_images.append(new_results[cut_index: index + 1])
                cut_index = index + 1
                curr_index = index + 1
        except:
            continue

    for index, i in enumerate(line_images):
        line_images[index] = sorted(i, key=lambda a: a[0][0] + a[0][1])
    texts = []
    for i in line_images:
        text = ''
        for index, j in enumerate(i):
            try:
                if index == len(i) - 1:
                    text += j[1]
                elif abs(i[index + 1][0][6] - i[index][0][6]) > 3 * (
                        abs(i[index][0][6] - i[index][0][0]) / len(i[index][1])):
                    text += j[1] + ' '
                else:
                    text += j[1]
            except:
                continue
        texts.append([[i[0][0], i[-1][0]], text])

    crop_w = crop_img.shape[1]
    document = layout(texts, crop_w, table_infos)
    # print(document)


    # for i in texts:
    #     print(11111, i)
    #     document += i[1] + '\n'
    if document == '':
        # document = rec_txt(np.array(ori_img.convert('L'))).replace('“', '').replace('‘', '')
        # torch.cuda.empty_cache()
        try:
            # _, img = cv2.imencode('.jpg', ori_img)
            # img = base64.b64encode(img.tostring())
            # data = {'img': img}
            # content = requests.post('http://172.30.81.191:32010/predict', data=data).json()[:2]
            # document = content
            content, _ = predict(Image.fromarray(ori_img).convert('L'))
            for indexi, i in enumerate(content[1]):
                if i[0] > 0.9:
                    content[0][indexi] = content[0][indexi][0]
                    content[1].pop(indexi)
            document = calculate(content)
            # torch.cuda.empty_cache()
        except:
            pass
    print('ddddddddddddddd', document)
    if document == ([], []):
        document = ''
    ori_end_time = datetime.now()
    ori_return = json.dumps([document])
    print('ori_time;', time.time() - ori_time, '\n', 'ori_start_time:', ori_start_time, '\n', 'ori_end_time:', ori_end_time)
    return response.text(ori_return)


@app.route('/check')
def health_check(request):
    return 'ok'


if __name__ == '__main__':
    app.config.KEEP_ALIVE = False
    app.config.RESPONSE_TIMEOUT = 7200
    app.run(host='0.0.0.0', port=8003)
