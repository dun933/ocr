import cv2
import math
import numpy as np
from PIL import Image
# from line_extract import test1 # import line_detect
# from Rec_text import rec_txt
import os

from pan.predict import text_predict
# from ctpn.ctpn_blstm_test_full import text_predict
import torch
from crnn_torch.model1 import predict
# from densent_ocr.model import predict
# from crnn_seq2seq_ocr.inference import attention
from viterbi import calculate
import time
import traceback
from docx.oxml.shared import OxmlElement, qn

COUNT = 1


def table_lines(ori_src, shape):
    shape = ori_src.shape
    # Image.fromarray(ori_src).save('123.jpg')
    # ori_src = test1.line_detect(ori_src)
    # Image.fromarray(ori_src).save('1234.jpg')
    src = Image.fromarray(ori_src)
    # src.thumbnail((700, 700), Image.ANTIALIAS)
    img_scale = 1.0
    src.thumbnail((int(shape[1]*img_scale), int(shape[0]*img_scale)), Image.ANTIALIAS)
    src = np.array(src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Image.fromarray(gray).show()
    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal = thresh
    vertical = thresh

    scale = 15

    horizontalsize = int(shape[0] / scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, iterations=1)
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=1)
    horizontal = cv2.blur(horizontal, (3, 3))

    verticalsize = int(shape[1] / scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    vertical = cv2.blur(vertical, (3, 3))
    # Image.fromarray(vertical + horizontal).show()

    # 霍夫直线变换,检测倾斜角度，进行旋转
    edges = cv2.Canny(vertical, 30, 240)
    lines_v = cv2.HoughLinesP(edges, 2, np.pi / 180, 200, minLineLength=20, maxLineGap=100)
    degree = 0
    if lines_v is not None:
        y_x_v = [(i[0][3] - i[0][1]) / (i[0][2] - i[0][0]) for i in lines_v]
        y_x_v = [i for i in y_x_v if i > 1 or i < -1 and i != np.inf and i != -np.inf]
        _y_x_v = [i for i in y_x_v if i >= 0]
        __y_x_v = [i for i in y_x_v if i <= 0]
        if len(_y_x_v) >= len(__y_x_v):
            y_x_v = _y_x_v
        else:
            y_x_v = __y_x_v

        degree_v = np.arctan(np.mean(y_x_v))
        degree_v = math.degrees(degree_v)
        degree = degree_v

        if degree > 0:
            degree = -90 + degree
        else:
            degree = 90 + degree
    print('table_degree_rotate', degree)

    if -10 < degree < 10:
        horizontal = np.array(Image.fromarray(horizontal).rotate(degree))
        vertical = np.array(Image.fromarray(vertical).rotate(degree))
        src = np.array(Image.fromarray(src).rotate(degree))
        ori_src = np.array(Image.fromarray(src).rotate(degree))

    # 判断是否加线
    x, y, w, h = cv2.boundingRect(horizontal)
    oints = cv2.bitwise_and(horizontal, vertical)
    wi = cv2.findNonZero(oints)
    # Image.fromarray(horizontal).show()
    xx = [i[0][0] for i in wi]
    yy = [i[0][1] for i in wi]
    if min(xx) > 0.05*w:
        cv2.line(vertical, (3, 3), (3, h - 3), (255, 255, 255), 1)
    if max(xx) < 0.95*w:
        cv2.line(vertical, (w - 3, y + 3), (x + w - 3, y + h - 3), (255, 255, 255), 1)
    if min(yy) > 0.05*h:
        cv2.line(horizontal, (3, 3), (w - 3, 3), (255, 255, 255), 1)
    if max(yy) < 0.95*h:
        cv2.line(horizontal, (3, h - 3), (w - 3, h - 3), (255, 255, 255), 1)

    mask = horizontal + vertical
    # Image.fromarray(mask).show()
    # import random
    # Image.fromarray(mask).save(f'1111111{random.randint(0, 100)}.jpg')

    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(~mask, kernel=kernel)
    erode_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size), (-1, -1))
    mask = cv2.morphologyEx(~mask, cv2.MORPH_OPEN, element)
    mask = ~mask
    # Image.fromarray(mask).show()

    # print(horizontal.shape, vertical.shape)
    joints = cv2.bitwise_and(horizontal, vertical)
    # Image.fromarray(joints).show()
    # x, y, w, h = cv2.boundingRect(mask)
    # Image.fromarray(mask[y:y+h, x:x+w]).show()
    # cv2.rectangle(mask, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 1)
    mask = cv2.blur(mask, (2, 2))
    Image.fromarray(mask).show()
    if not joints.any():
        return False

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [''] * len(contours)
    boundRect = [''] * len(contours)
    rois = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 100:
            continue
        contours_poly[i] = cv2.approxPolyDP(np.array(contours[i]), 3, True)
        boundRect[i] = cv2.boundingRect(np.array(contours_poly[i]))
        # boundRect[i] = cv2.boundingRect(np.array(contours[i]))

        roi = joints[boundRect[i][1]:boundRect[i][1] + boundRect[i][3],
              boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]

        joints_contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(joints_contours)
        # if len(joints_contours) < 1:
        #     print(111)
        #     continue

        ytt = ori_src[int(boundRect[i][1]/img_scale):int(boundRect[i][1]/img_scale) + int(boundRect[i][3]/img_scale),
              int(boundRect[i][0]/img_scale):int(boundRect[i][0]/img_scale) + int(boundRect[i][2]/img_scale)]
        # Image.fromarray(ytt).show()
        rois.append([ytt, [int(b/img_scale) for b in boundRect[i]]])
        # Image.fromarray(ytt).show()
        # cv2.drawContours(src, [contours[i]], -1, (0, 255, 255), 1)
        # Image.fromarray(src).show()

    new_con, _ = cv2.findContours(joints, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_x = []
    num_y = []
    for i in new_con:
        num_x.append(cv2.minEnclosingCircle(i)[0][0])
        num_y.append(cv2.minEnclosingCircle(i)[0][1])
    num_x = sorted(num_x)
    num_y = sorted(num_y)
    # print(5555555555555, len(num_y))
    xs = set()
    for index in range(len(num_x) - 1):
        if abs(num_x[index] - num_x[index + 1]) < 30:
            num_x[index + 1] = num_x[index]
            xs.add(num_x[index])
    ys = set()
    for index in range(len(num_y) - 1):
        if abs(num_y[index] - num_y[index + 1]) < 30:
            num_y[index + 1] = num_y[index]
            ys.add(num_y[index])
    # print(111, mask.shape, tuple([int(s/img_scale) for s in mask.shape]))
    return len(xs) - 1, len(ys) - 1, list(xs), list(ys), rois, tuple([int(s/img_scale) for s in mask.shape])


def draw_line(image):
    src = image
    if not src.data:
        print('not picture')
    src_height, src_width = src.shape[:2]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # canny = cv2.Canny(gray, 80, 120)
    # # Image.fromarray(canny).show()
    # kernel = np.ones((2, 2), np.uint8)
    # gray = cv2.erode(gray, kernel=kernel)
    # Image.fromarray(gray).show()

    # erode_size = int(src_height / 300)
    # if erode_size % 2 == 0:
    #     erode_size += 1
    erode_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size), (-1, -1))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, element)
    # Image.fromarray(gray).show()
    # erod = cv2.erode(gray, element)
    # blur_size = int(src_height / 200)
    # if blur_size % 2 == 0:
    #     blur_size += 1
    # blur = cv2.GaussianBlur(erod, (blur_size, blur_size), 0, 0)

    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -10)
    # Image.fromarray(thresh).show()
    horizontal = thresh
    vertical = thresh

    scale = 20

    # print(1111111111111, horizontal.shape)
    horizontalsize = int(horizontal.shape[0] / scale)
    print(222222222, horizontalsize)

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.blur(horizontal, (2, 2))
    # Image.fromarray(horizontal).show()
    # horizontal = cv2.dilate(horizontal, horizontalStructure, (20, 20))

    verticalsize = int(vertical.shape[1] / scale)
    print(verticalsize)

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    vertical = cv2.blur(vertical, (2, 2))
    # Image.fromarray(vertical).show()

    mask = horizontal + vertical
    # Image.fromarray(mask).show()
    # TODO
    # erode_size = 5
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size), (-1, -1))
    # mask = cv2.morphologyEx(~mask, cv2.MORPH_OPEN, element)
    # print(111111111111)
    # Image.fromarray(mask).show()



    # x, y, w, h = cv2.boundingRect(mask)
    # Image.fromarray(mask[y:y+h, x:x+w]).show()
    # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # mask = cv2.blur(mask, (5, 5))
    joints = cv2.bitwise_and(horizontal, vertical)
    # print(joints)
    # print(horizontal)
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.dilate(mask, kernel=kernel)
    # Image.fromarray(joints).show()
    if not joints.any():
        return [], []

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 判断交点数，小于2则不为表格
    num_x = []
    for i in contours:
        num_x.append(cv2.minEnclosingCircle(i)[0][0])
    num_x = sorted(num_x)
    xs = set()
    for index in range(len(num_x) - 1):
        if abs(num_x[index] - num_x[index + 1]) < 5:
            num_x[index + 1] = num_x[index]
            xs.add(num_x[index])
    # print('xs', len(xs))
    if len(xs) <= 1:
        return [], []
    # for i in range(len(contours)):
    #     import random
    #     cv2.drawContours(src, [contours[i]], -1, (0, random.randint(0, 255), 0), 3)
    # Image.fromarray(src).show()
    # print(22222222, joints.shape)
    contours_poly = [''] * len(contours)
    boundRect = [''] * len(contours)
    rois = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 100:
            continue
        contours_poly[i] = cv2.approxPolyDP(np.array(contours[i]), 3, True)
        boundRect[i] = cv2.boundingRect(np.array(contours_poly[i]))
        # boundRect[i] = cv2.boundingRect(np.array(contours[i]))

        roi = joints[boundRect[i][1]:boundRect[i][1] + boundRect[i][3],
              boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]

        joints_contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(joints_contours)
        if len(joints_contours) < 1:
            continue

        ytt = src[boundRect[i][1]:boundRect[i][1] + boundRect[i][3], boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]
        # Image.fromarray(ytt).save('data/{}.jpg'.format(i))
        # print(boundRect[i])
        # Image.fromarray(ytt).show()
        # rois.append(src(boundRect[i]).clone())
        rois.append([ytt, list(boundRect[i])])
        # Image.fromarray(ytt).show()
        # cv2.rectangle(src, (boundRect[i][0], boundRect[i][1]), (boundRect[i][0] + boundRect[i][2], boundRect[i][1] + boundRect[i][3]), (0, 255, 0), 3)
    return rois, src


def extract_table(image):
    st_time = time.time()
    rois, src = draw_line(image)
    print('draw_time is ', time.time()-st_time)
    if not rois:
        return []
    sort_table = sorted(rois, key=lambda i: i[1][3], reverse=True)
    tables = [sort_table[0]]
    # Image.fromarray(tables[0][0]).show()
    for i in sort_table[1:]:
        count = 0
        for j in tables:
            # Image.fromarray(j[0]).show()
            if j[1][1] < i[1][1] + 10 and i[1][1] - 10 < j[1][1] + j[1][3]:
                continue
            else:
                count += 1
        if count == len(tables):
            tables.append(i)
    # for i in tables:
    #     Image.fromarray(i[0]).show()
    cells = []
    try:
        for i in tables:
            # i_h, i_w = i[0].shape[:-1]
            # cv2.rectangle(i[0], (1, 1), (i_w - 1, i_h - 1), (0, 0, 0), 2)
            cols, rows, col_point, row_point, tables, table_shape = table_lines(i[0], image.shape)
            print('cols and rows', cols, rows)
            # Image.fromarray(i[0]).show()
            if cols > 1 and rows > 1:
                cells.append([i, cols, rows, col_point, row_point, tables, table_shape])
    except Exception as e:
        print('extract_table_error', e)
    # for index, i in enumerate(cells[0][5]):
    #     Image.fromarray(i[0]).show()
    # print(55555555, len(cells))
    if cells:
        return cells
    else:
        return 'not table'
    # generate_table(document, cols, rows, tables)
    # Image.fromarray(src).save('space_d.jpg')


# extract_table(cv2.imread('table.jpg'))
def generate_table(cell, src):
    # import pickle
    pos, cols, rows, col_point, row_point, tables, table_shape = cell[0][1], cell[1], cell[2], cell[3], cell[4], cell[5], cell[6]
    col_point = sorted(col_point)
    row_point = sorted(row_point)
    tables = sorted(tables, key=lambda i: i[1][3])[:-1]
    tables = sorted(tables, key=lambda i: i[1][0] + i[1][1])

    # 表格内所有单字位置
    table_im = src[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]
    table_line_regions = text_predict(table_im, 1, 1, table_im)
    torch.cuda.empty_cache()

    word_list = []
    # print('table_line_length', len(table_line_regions))
    for region_index, region in enumerate(table_line_regions):
        region_y = [region[0][1], region[0][5]]
        region_x = [region[0][0], region[0][2]]
        # Image.fromarray(region[1]).save(f'1/{region_index}.jpg')
        content = predict(Image.fromarray(region[1]).convert('L'))

        torch.cuda.empty_cache()
        content = (content[0][0], content[0][1], content[1])
        for indexi, cont in enumerate(content[1]):
            if cont[0] > 0.9:
                content[0][indexi] = content[0][indexi][0]
                content[1][indexi] = [-1]
        while 1:
            try:
                content[1].remove([-1])
            except:
                break
        x = content[2]
        content = calculate(content)

        for index, word in enumerate(content):
            word_list.append(
                [[x[index][0] + region_x[0], region_y[0], x[index][1] + region_x[0], region_y[0], x[index][0]
                  + region_x[0], region_y[1], x[index][1] + region_x[0], region_y[1]], word])

    # # 保存表格行列焦点坐标
    # show_im = np.ones(table_shape, np.uint8)
    # import itertools
    # for x, y in itertools.product([int(i) for i in col_point], [int(i) for i in row_point]):
    #     cv2.circle(show_im, (x, y), 1, (255, 255, 255), 1)
    # Image.fromarray(show_im).save('show_im.jpg')

    for i in tables:
        d = {'col_begin': 0, 'col_end': 0, 'row_begin': 0, 'row_end': 0}
        for index, value in enumerate(col_point):
            if index == 0:
                d_range = 50
            else:
                d_range = (col_point[index] - col_point[index - 1]) / 2
            if i[1][0] > col_point[index] - d_range:
                d['col_begin'] = index
        for index, value in enumerate(col_point):
            if index == len(col_point) - 1:
                d_range = 50
            else:
                d_range = (col_point[index + 1] - col_point[index]) / 2
            if i[1][0] + i[1][2] < col_point[index] + d_range:
                d['col_end'] = index
                break
        for index, value in enumerate(row_point):
            if index == 0:
                d_range = 50
            else:
                d_range = (row_point[index] - row_point[index - 1]) / 2
            if i[1][1] > row_point[index] - d_range:
                d['row_begin'] = index
        for index, value in enumerate(row_point):
            if index == len(row_point) - 1:
                d_range = 50
            else:
                d_range = (row_point[index + 1] - row_point[index]) / 2
            if i[1][1] + i[1][3] < row_point[index] + d_range:
                d['row_end'] = index
                break
        if d['col_begin'] >= d['col_end']:
            d['col_end'] = d['col_begin'] + 1
        if d['row_begin'] >= d['row_end']:
            d['row_end'] = d['row_begin'] + 1
        # print('123'*3, d)
        i.append(d)

    # print(pos[0], pos[1], pos[2], pos[3])
    # table_im = src[pos[1]:pos[1]+pos[3], pos[0]:pos[0]+pos[2]]
    # Image.fromarray(table_im).show()
    # images = text_predict(table_im, 1, 1, table_im)

    cell_list = []
    for row_p in range(len(row_point) - 1):
        for col_p in range(len(col_point) - 1):
            roi = table_im[int(row_point[row_p]):int(row_point[row_p + 1]), int(col_point[col_p]):int(col_point[col_p + 1])]
            cell_list.append([roi, [int(col_point[col_p]), int(row_point[row_p]), int(col_point[col_p+1]-col_point[col_p]), int(row_point[row_p+1]-int(row_point[row_p]))],
                              {'col_begin':col_p, 'col_end':col_p+1, 'row_begin':row_p, 'row_end':row_p+1}, 0])

    # 判断单元格是否正确检测
    for i in tables:
        col_begin, col_end, row_begin, row_end = \
            i[-1]['col_begin'], i[-1]['col_end'], i[-1]['row_begin'], i[-1]['row_end']
        for col in range(col_begin, col_end):
            for row in range(row_begin, row_end):
                for cell in cell_list:
                    if cell[2]['col_begin'] == col_begin and cell[2]['col_end'] == col_end and\
                            cell[2]['row_begin'] == row_begin and cell[2]['row_end'] == row_end:
                        cell[-1] = 1
    # 没有检测到单元格则赋值
    for i in cell_list:
        if i[-1] == 0:
            print('not detect cell', i[1:])
            tables.append(i[:-1])

    # images = text_predict(table_im)

    # # 单元格位置
    # # for cell in tables:
    # #     print(cell[1:])
    # # 保存表格图
    # save_table = table_im.copy()
    # # for word in word_list:
    # #     word = word[0]
    # #     cv2.rectangle(save_table, (word[0], word[1]), (word[6], word[7]), (255, 0, 0), 1)
    # for i in table_line_regions:
    #     print(123456, i[0])
    #     cv2.rectangle(save_table, (i[0][0] - 1, i[0][1] - 1), (i[0][6] + 1, i[0][7] + 1), (255, 0, 0), 1)
    # # import random
    # # for i in tables:
    # #     cv2.rectangle(save_table, (i[1][0], i[1][1]), (i[1][0]+i[1][2], i[1][1]+i[1][3]), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)
    # from config_url import DETECT_URL
    # import requests, base64
    # _, img = cv2.imencode('.jpg', table_im)
    # img = base64.b64encode(img.tostring())
    # # data = {'img': img, 'scale_w': scale_w, 'scale_h': scale_h, 'ori_img': ori}
    # data = {'img': img, 'scale_w': 1, 'scale_h': 1, 'ori_img': img}
    # crop_area_json = requests.post(DETECT_URL, data=data)
    # crop_area = []
    # # while_i += 1
    # if crop_area_json.json() != '':
    #     for i in crop_area_json.json():
    #         image = base64.b64decode(i[1])
    #         image = np.fromstring(image, np.uint8)
    #         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #         crop_area.append([i[0], image])
    # for te in crop_area:
    #     print(2221111, te[0])
    #     t = te[0]
    #     cv2.rectangle(save_table, (t[0], t[1]), (t[6], t[7]), (0, 0, 255), 1)
    # Image.fromarray(save_table).save('able1.jpg')
    # Image.fromarray(table_im).save('able3.jpg')


    # 去除检测错误的表格单元格
    tables_cell = {}
    for cell in tables:
        tmp = f"{cell[2]['row_begin']}_{cell[2]['row_end']}_{cell[2]['col_begin']}_{cell[2]['col_end']}"
        if tmp not in tables_cell.keys():
            tables_cell[tmp] = cell[:-1]
        else:
            if tables_cell[tmp][1][2]*tables_cell[tmp][1][3] < cell[1][2]*cell[1][3]:
                tables_cell[tmp] = cell[:-1]
    # for cell in tables_cell:
    #     print(111, cell[1:])
    tables = [[v[0], v[1], {'row_begin': int(k.split('_')[0]), 'row_end': int(k.split('_')[1]), 'col_begin': int(k.split('_')[2]),
                            'col_end': int(k.split('_')[3])}] for k, v in tables_cell.items()]

    save_table = table_im.copy()
    for index_i, i in enumerate(tables):
        print('cell location: ', i[-1])
        cell_region = [i[1][0], i[1][1], i[1][0] + i[1][2], i[1][1] + i[1][3]]
        cv2.putText(save_table, str(index_i), (cell_region[0]+2, cell_region[1]+2), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        cv2.rectangle(save_table, (cell_region[0], cell_region[1]), (cell_region[2], cell_region[3]), (255, 0, 0), 1)
        word_str = []

        for word in word_list:
            word_center_point = ((word[0][0] + word[0][2]) / 2, (word[0][1] + word[0][5]) / 2)
            if cell_region[0] < word_center_point[0] < cell_region[2] and cell_region[1] < word_center_point[1] < \
                    cell_region[3]:
                word_str.append(word)
        # if i[2]['row_begin'] == 3 and i[2]['row_end'] == 4 and i[2]['col_begin'] == 0 and i[2]['col_end'] == 1:
        #     print(cell_region)
        #     print(word_str)
        word_str = sorted(word_str, key=lambda x: x[0][1])
        # print('word_str', word_str)
        # print('table', i[2])
        # print(i[2], word_str)
        word_lines = []
        word_temp = []
        for index, word in enumerate(word_str):
            if len(word_temp) == 0:
                word_temp.append(word)
                if len(word_str) == 1:
                    word_lines.append(word_temp)
                continue
            if word[0][1] == word_temp[-1][0][1]:
                word_temp.append(word)
            else:
                word_temp = sorted(word_temp, key=lambda x: x[0][0])
                # print(1111, word_temp)
                word_lines.append(word_temp)
                word_temp = [word]
            if index == len(word_str) - 1:
                if len(word_temp) != 0:
                    # print(2222, word_temp)
                    word_lines.append(word_temp)
        word_str = ''
        # new_word_lines = []
        # for line in word_lines:
        #     if line in new_word_lines:
        #         print(1111111)
        #         continue
        #     new_word_lines.append(line)
        # word_lines = new_word_lines.copy()
        for line in word_lines:
            # print('line', line)
            for word in line:
                word_str += word[1]
        i.append([word_str, i[1][2], i[1][3]])
    Image.fromarray(save_table).save('able1.jpg')
    # for cell in tables:
    #     # print('*'*5, cell[1:])
    #     cell_w, cell_h = cell[1][2], cell[1][3]
    #     cell_ims, text = [], ''
    #     for image in images:
    #         image_im, cell_im = image[0], cell[1]
    #         if image_im[0] > cell_im[0]+cell_im[2]:
    #             continue
    #         if image_im[1] > cell_im[1]+cell_im[3]:
    #             continue
    #         if image_im[6] < cell_im[0]:
    #             continue
    #         if image_im[7] < cell_im[1]:
    #             continue
    #         x0, y0, x1, y1 = max(image_im[0], cell_im[0]), max(image_im[1], cell_im[1]), \
    #                          min(image_im[6], cell_im[0]+cell_im[2]), min(image_im[7], cell_im[1]+cell_im[3])
    #         cell_ims.append([x0, y0, x1, y1])
    #     for i in cell_ims:
    #         try:
    #             cell_im = table_im[i[1]:i[3], i[0]:i[2]]
    #             content = predict(Image.fromarray(cell_im).convert('L'))
    #             for indexi, i in enumerate(content[1]):
    #                 if i[0] > 0.9:
    #                     content[0][indexi] = content[0][indexi][0]
    #                     content[1][indexi] = [-1]
    #             while 1:
    #                 try:
    #                     content[1].remove([-1])
    #                 except:
    #                     break
    #             content = calculate(content)
    #             # Image.fromarray(j[1]).save('found/{}.jpg'.format(''.join(img_path.split('/'))))
    #             torch.cuda.empty_cache()
    #             text += content
    #         except Exception as ex:
    #             print('ocr error', ex)
    #             continue
    #     cell.append([text, cell_w, cell_h])
    #     print('cell text:', text)

    tables = sorted(tables, key=lambda x: x[2]['row_begin'])

    new_table = []
    for i in tables:
        new_table.append([i[2], i[3]])

    return new_table, rows, cols, pos


if __name__ == '__main__':
    import fitz
    img = Image.open(r'11.jpg').convert('RGB')
    # img.thumbnail((600, 600), Image.ANTIALIAS)
    # pdf_path = r'C:\Users\Admin\Desktop\table_test.jpg'
    # pdf = fitz.open(pdf_path)
    # page_fitz = pdf[73]
    # trans = fitz.Matrix(3, 3).preRotate(0)
    # pm = page_fitz.getPixmap(matrix=trans, alpha=False)
    # img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)

    # img.show()
    # img_name = img_name.convert('RGB')
    img = np.array(img)



    a = extract_table(img)
    for aa in a:
        i = ['table', generate_table(aa, img)]
        print(i)
    # i = [i, 'table', 10.5, 1, 0]
    # i = [i[0][1], i[1], i[2], i[3], i[4]]
    #
    # from restore_table import restore_table
    # from docx import Document
    # doc = Document()
    # doc = restore_table(doc, i, Image.fromarray(img))
    # doc.save(pdf_path.replace('.pdf', '.docx'))

