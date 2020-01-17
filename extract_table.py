import cv2
import math
import numpy as np
from PIL import Image

# from Rec_text import rec_txt
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
# from pan.predict import text_predict
# from crnn_torch.model1 import predict
# from ctpn.ctpn_blstm_test_full import text_predict
# from densent_ocr.model import predict
# from crnn_seq2seq_ocr.inference import attention
from viterbi import calculate
from config_url import DETECT_URL, RECOGNISE_URL
import base64
import requests


COUNT = 1


def table_lines(src, shape):
    # src_height, src_width = src.shape[:2]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

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

    # 判断是否加线
    x, y, w, h = cv2.boundingRect(horizontal)
    oints = cv2.bitwise_and(horizontal, vertical)
    wi = cv2.findNonZero(oints)
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
    mask = cv2.blur(mask, (5, 5))
    # Image.fromarray(mask).show()
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

        # ytt = src[boundRect[i][1]:boundRect[i][1] + boundRect[i][3], boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]
        # scale = src.shape[0]/ori_image.shape[0]
        ytt = src[boundRect[i][1]:boundRect[i][1] + boundRect[i][3], boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]

        # rois.append(src(boundRect[i]).clone())
        rois.append([ytt, list(boundRect[i])])
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
    return len(xs) - 1, len(ys) - 1, list(xs), list(ys), rois


def draw_line(src):
    if not src.data:
        print('not picture')
    src_height, src_width = src.shape[:2]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    erode_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size), (-1, -1))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, element)

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

    mask = horizontal + vertical

    joints = cv2.bitwise_and(horizontal, vertical)
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
        if len(joints_contours) < 1:
            continue

        ytt = src[boundRect[i][1]:boundRect[i][1] + boundRect[i][3], boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]

        rois.append([ytt, list(boundRect[i])])
    return rois, src


def extract_table(image):
    # image = Image.fromarray(ori_image).copy()
    # image.thumbnail((600, 600), Image.ANTIALIAS)
    # image = ori_image.copy()
    print(image.shape)
    # image = np.array(image)
    # scale = image.shape[0]/ori_image.shape[0]
    rois, src = draw_line(image)
    print(7777777777, image.shape)
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
            # print(i)
            cols, rows, col_point, row_point, tables = table_lines(i[0], image.shape)
            print('cols and rows', cols, rows)
            # Image.fromarray(i[0]).show()
            if cols > 1 and rows > 1:
                cells.append([i, cols, rows, col_point, row_point, tables])
    except Exception as e:
        print(111111111111234)
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


import traceback
from docx.oxml.shared import OxmlElement, qn


def set_cell_vertical_alignment(cell, align="center"):
    try:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcValign = OxmlElement('w:vAlign')
        tcValign.set(qn('w:val'), align)
        tcPr.append(tcValign)
        return True
    except:
        traceback.print_exc()
        return False


# extract_table(cv2.imread('table.jpg'))
def generate_table(cell, ori_img):
    # import pickle
    # pickle.dump(cell, open('table.pkl', 'wb'))
    pos, cols, rows, col_point, row_point, tables = cell[0][1], cell[1], cell[2], cell[3], cell[4], cell[5]
    print(11111111111111, pos)
    table_im = ori_img[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]
    # table_line_regions = text_predict(table_im, 1, 1, table_im)
    table_line_regions = []
    _, img = cv2.imencode('.jpg', table_im)
    img = base64.b64encode(img.tostring())
    data = {'img': img, 'scale_w': 1, 'scale_h': 1, 'ori_img': img}
    images_json = requests.post(DETECT_URL, data=data)
    if images_json.json() != '':
        for i in images_json.json():
            image = base64.b64decode(i[1])
            image = np.fromstring(image, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            table_line_regions.append([i[0], image])

    torch.cuda.empty_cache()
    word_list = []
    for region in table_line_regions:
        region_y = [region[0][1], region[0][5]]
        region_x = [region[0][0], region[0][2]]
        # content = predict(Image.fromarray(region[1]).convert('L'))
        # torch.cuda.empty_cache()
        # content = (content[0][0], content[0][1], content[1])
        _, img = cv2.imencode('.jpg', region[1])
        img = base64.b64encode(img.tostring())
        data = {'img': img}
        contents = requests.post(RECOGNISE_URL, data=data).json()
        content, x = contents[:2], contents[2]
        for indexi, cont in enumerate(content[1]):
            if cont[0] > 0.9:
                content[0][indexi] = content[0][indexi][0]
                content[1][indexi] = [-1]
        while 1:
            try:
                content[1].remove([-1])
            except:
                break
        # x = content[2]
        content = calculate(content)
        for index, word in enumerate(content):
            word_list.append(
                [[x[index][0] + region_x[0], region_y[0], x[index][1] + region_x[0], region_y[0], x[index][0]
                  + region_x[0], region_y[1], x[index][1] + region_x[0], region_y[1]], word])
    # print(word_list)
    # for region in table_line_regions:
    #     cv2.rectangle(table_im, (region[0][0], region[0][1]), (region[0][6], region[0][7]), (0, 255, 0), 1)
    # for i in word_list:
    #     cv2.rectangle(table_im, (i[0][0], i[0][1]), (i[0][6], i[0][7]), (255, 0, 0), 1)
    #
    # Image.fromarray(table_im).save('single_word.jpg')

    col_point = sorted(col_point)
    row_point = sorted(row_point)
    tables = sorted(tables, key=lambda i: i[1][3])[:-1]
    tables = sorted(tables, key=lambda i: i[1][0] + i[1][1])

    for i in tables:
        d = {'col_begin': 0, 'col_end': 0, 'row_begin': 0, 'row_end': 0}
        for index, value in enumerate(col_point):
            if index == 0:
                d_range = 50
            else:
                d_range = (col_point[index] - col_point[index - 1]) / 2
            if i[1][0] > col_point[index] - d_range:
                # print(33333333333, i[1], index)
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
        i.append(d)

    # for index, i in enumerate(tables):
    #     texts = ''
    #     try:
    #         print(11111111, i[1:])
    #         i[0] = Image.fromarray(i[0])
    #         new_i = i[0].copy()
    #         ori_w, ori_h = i[0].size
    #         new_i.thumbnail((1500, 1500), Image.ANTIALIAS)
    #         scale_w, scale_h = new_i.size
    #         scale_w, scale_h = ori_w / scale_w, ori_h / scale_h
    #         new_i = np.array(new_i.convert('RGB'))
    #         # print(type(new_i))
    #         # Image.fromarray(new_i).save('core.jpg')
    #         if new_i.shape[1] > 16 and new_i.shape[0] > 16:
    #             images = text_predict(new_i, scale_w, scale_h, np.array(i[0]))
    #             torch.cuda.empty_cache()
    #             # images = text_predict(new_i)
    #         else:
    #             i.append([texts, i[1][2], i[1][3]])
    #             continue
    #         # torch.cuda.empty_cache()
    #         if images:
    #             for image in sorted(images, key=lambda ii: ii[0][1]):
    #                 content = predict(Image.fromarray(image[1]).convert('L'))
    #                 for indexi, cont in enumerate(content[1]):
    #                     if cont[0] > 0.9:
    #                         content[0][indexi] = content[0][indexi][0]
    #                         content[1][indexi] = [-1]
    #                 while 1:
    #                     try:
    #                         content[1].remove([-1])
    #                     except:
    #                         break
    #                 content = calculate(content)
    #                 # print('content', content)
    #                 texts += content
    #
    #         elif new_i.any() and new_i.shape[0] < new_i.shape[1] * 1.5:
    #             try:
    #                 content = predict(Image.fromarray(new_i).convert('L'))
    #                 for indexi, cont in enumerate(content[1]):
    #                     if cont[0] > 0.9:
    #                         content[0][indexi] = content[0][indexi][0]
    #                         content[1][indexi] = [-1]
    #                 while 1:
    #                     try:
    #                         content[1].remove([-1])
    #                     except:
    #                         break
    #                 content = calculate(content)
    #                 texts += content
    #             except Exception as ex:
    #                 print('small_image_warning', ex)
    #         i.append([texts, i[1][2], i[1][3]])
    #         # print('54321')
    #     except Exception as e:
    #         print('table_text warning', e)

    for i in tables:
        cell_region = [i[1][0], i[1][1], i[1][0] + i[1][2], i[1][1] + i[1][3]]
        word_str = []
        for word in word_list:
            word_center_point = ((word[0][0] + word[0][2]) / 2, (word[0][1] + word[0][5]) / 2)
            if cell_region[0] < word_center_point[0] < cell_region[2] and cell_region[1] < word_center_point[1] < \
                    cell_region[3]:
                word_str.append(word)
        word_str = sorted(word_str, key=lambda x: x[0][1])
        word_lines = []
        word_temp = []
        for index, word in enumerate(word_str):
            if len(word_temp) == 0:
                word_temp.append(word)
                continue
            if word[0][1] == word_temp[-1][0][1]:
                word_temp.append(word)
            else:
                word_temp = sorted(word_temp, key=lambda x: x[0][0])
                word_lines.append(word_temp)
                word_temp = [word]
            if index == len(word_str) - 1:
                if len(word_temp) != 0:
                    word_lines.append(word_temp)
        word_str = ''
        for line in word_lines:
            for word in line:
                word_str += word[1]
        i.append([word_str, i[1][2], i[1][3]])

    new_table = []
    for i in tables:
        new_table.append([i[2], i[3]])

    return new_table, rows, cols, pos


if __name__ == '__main__':
    import fitz
    # img = Image.open(r'C:\Users\Admin\Desktop\20120822 第三次变更__21.jpg')
    # img_name.thumbnail((595.5, 842.25), Image.ANTIALIAS)
    pdf_path = r'C:\Users\Admin\Desktop\3 无锡博翱 内档—第二次变更.pdf'
    pdf = fitz.open(pdf_path)
    page_fitz = pdf[73]
    trans = fitz.Matrix(3, 3).preRotate(0)
    pm = page_fitz.getPixmap(matrix=trans, alpha=False)
    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)

    # img.show()
    # img_name = img_name.convert('RGB')
    img = np.array(img)

    a = extract_table(img)
    i = ['table', generate_table(a[0])]
    print(i)
    i = [i, 'table', 10.5, 1, 0]
    i = [i[0][1], i[1], i[2], i[3], i[4]]

    from restore_table import restore_table
    from docx import Document
    doc = Document()
    doc = restore_table(doc, i, Image.fromarray(img))
    doc.save(pdf_path.replace('.pdf', '.docx'))

