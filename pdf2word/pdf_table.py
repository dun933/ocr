import random

import cv2
import math
import numpy as np
from PIL import Image


COUNT = 1


def table_lines(src, shape):
    src_height, src_width = src.shape[:2]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    erode_size = int(src_height / 300)
    if erode_size % 2 == 0:
        erode_size += 1
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size))
    erod = cv2.erode(gray, element)
    blur_size = int(src_height / 200)
    if blur_size % 2 == 0:
        blur_size += 1
    blur = cv2.GaussianBlur(erod, (blur_size, blur_size), 0, 0)

    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal = thresh
    vertical = thresh

    scale = 20

    # Image.fromarray(thresh).show()
    # print(1111111111111, horizontal.shape)
    horizontalsize = int(shape[0] / scale)
    # print(222222222, horizontalsize)

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.blur(horizontal, (3, 3))
    # Image.fromarray(horizontal).show()
    # horizontal = cv2.dilate(horizontal, horizontalStructure, (20, 20))

    verticalsize = int(shape[1] / scale)

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    # Image.fromarray(vertical).show()
    vertical = cv2.blur(vertical, (3, 3))

    # joints = cv2.bitwise_and(horizontal, vertical)

    mask = horizontal + vertical

    joints = cv2.bitwise_and(horizontal, vertical)
    # x, y, w, h = cv2.boundingRect(mask)
    # Image.fromarray(mask[y:y+h, x:x+w]).show()
    # cv2.rectangle(mask, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 1)
    mask = cv2.blur(mask, (5, 5))
    # print(horizontal)
    # joints = cv2.bitwise_xor(mask, joints)
    # Image.fromarray(mask).show()

    # 霍夫直线变换,检测倾斜角度，进行旋转
    lines = cv2.HoughLinesP(mask, 2, np.pi / 180, 200, minLineLength=40, maxLineGap=100)
    y_x = [abs((i[0][3] - i[0][1]) / (i[0][2] - i[0][0])) for i in lines]
    y_x = [i for i in y_x if i > 1]
    degree = np.arctan(np.mean(y_x))
    degree = 90 - math.degrees(degree)
    if degree < 5:
        mask = np.array(Image.fromarray(mask).rotate(degree))
        src = np.array(Image.fromarray(src).rotate(degree))

    Image.fromarray(mask).show()
    if not joints.any():
        return False

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(src, contours, -1, (0, 255, 0), 1)
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
        # if len(joints_contours) < 1:
        #     print(111)
        #     continue

        ytt = src[boundRect[i][1]:boundRect[i][1] + boundRect[i][3], boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]
        # Image.fromarray(ytt).save('test_image/{}_{}_{}.jpg'.format(i, boundRect[i][1], boundRect[i][0]))
        # Image.fromarray(ytt).save('test_image/{}.jpg'.format(i))

        # rois.append(src(boundRect[i]).clone())
        rois.append([ytt, list(boundRect[i])])

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
        if abs(num_x[index] - num_x[index + 1]) < 10:
            num_x[index + 1] = num_x[index]
            xs.add(num_x[index])
    ys = set()
    for index in range(len(num_y) - 1):
        if abs(num_y[index] - num_y[index + 1]) < 10:
            num_y[index + 1] = num_y[index]
            ys.add(num_y[index])
    return len(xs) - 1, len(ys) - 1, list(xs), list(ys), rois


def draw_line(image):
    src = image
    if not src.data:
        print('not picture')
    src_height, src_width = src.shape[:2]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    erode_size = int(src_height / 300)
    if erode_size % 2 == 0:
        erode_size += 1
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size))
    # erod = cv2.erode(gray, element)
    # blur_size = int(src_height / 200)
    # if blur_size % 2 == 0:
    #     blur_size += 1
    # blur = cv2.GaussianBlur(erod, (blur_size, blur_size), 0, 0)

    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # Image.fromarray(thresh).show()
    horizontal = thresh
    vertical = thresh

    scale = 20

    # print(1111111111111, horizontal.shape)
    horizontalsize = int(horizontal.shape[0] / scale)
    # print(222222222, horizontalsize)

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.blur(horizontal, (3, 3))
    # Image.fromarray(horizontal).show()
    # horizontal = cv2.dilate(horizontal, horizontalStructure, (20, 20))

    verticalsize = int(vertical.shape[1] / scale)

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    vertical = cv2.blur(vertical, (3, 3))
    # Image.fromarray(vertical).show()

    mask = horizontal + vertical
    # x, y, w, h = cv2.boundingRect(mask)
    # Image.fromarray(mask[y:y+h, x:x+w]).show()
    # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # mask = cv2.blur(mask, (5, 5))
    joints = cv2.bitwise_and(horizontal, vertical)
    # Image.fromarray(joints).show()
    # print(joints)
    # print(horizontal)
    # Image.fromarray(mask).show()
    if not joints.any():
        return [], []

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 判断交点数，小于2则不为表格
    num_x = []
    for i in contours:
        num_x.append(cv2.minEnclosingCircle(i)[0][0])
    num_x = sorted(num_x)
    # print(5555555555555, len(num_y))
    xs = set()
    for index in range(len(num_x) - 1):
        if abs(num_x[index] - num_x[index + 1]) < 10:
            num_x[index + 1] = num_x[index]
            xs.add(num_x[index])
    # print(len(xs))
    if len(xs) <= 1:
        return [], []
    # cv2.drawContours(src, contours, -1, (0, 255, 0), 1)
    # Image.fromarray(src).show()
    # print(22222222, joints.shape)
    # print(len(contours))
    contours_poly = [''] * len(contours)
    boundRect = [''] * len(contours)
    rois = []

    # print(11111111, len(contours))
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

        # rois.append(src(boundRect[i]).clone())
        rois.append([ytt, list(boundRect[i])])
        # cv2.rectangle(src, (boundRect[i][0], boundRect[i][1]), (boundRect[i][0] + boundRect[i][2], boundRect[i][1] + boundRect[i][3]), (0, 0, 0), 2)
    return rois, src


def extract_table(image):
    rois, src = draw_line(image)
    if not rois:
        return []
    # rois, src = draw_line(src)

    # Image.fromarray(src).show()
    sort_table = sorted(rois, key=lambda i: i[1][3], reverse=True)

    # im_white = np.zeros((842, 595, 3), np.uint8)
    # im_white = 255 - im_white
    tables = [sort_table[0]]
    for i in sort_table[1:]:
        # cv2.rectangle(im_white, (i[1][0], i[1][1]), (i[1][0] + i[1][2], i[1][1] + i[1][3]), (255, 0, 0), 1)
        count = 0
        # print(len(tables))
        for j in tables:
            # cv2.rectangle(im_white, (j[1][0], j[1][1]), (j[1][0] + j[1][2], j[1][1] + j[1][3]), (0, 0, 0), 1)
            if j[1][1] < i[1][1] + 50 and i[1][1] - 50 < j[1][1] + j[1][3]:
                continue
            else:
                count += 1
        if count == len(tables):
            tables.append(i)
    # Image.fromarray(im_white).show()
    cells = []

    # print('table_len', len(tables))
    try:
        for i in tables:
            # print(i[1])
            # cv2.rectangle(im_white, (i[1][0], i[1][1]), (i[1][0]+i[1][2], i[1][1]+i[1][3]), (255, 0, 0), 1)
            # Image.fromarray(i[0]).show()
            # i_h, i_w = i[0].shape[:-1]
            # cv2.rectangle(i[0], (1, 1), (i_w - 1, i_h - 1), (0, 0, 0), 2)
            cols, rows, col_point, row_point, tables = table_lines(i[0], image.shape)
            # Image.fromarray(i[0]).show()
            if cols > 1 or rows > 1:
                cells.append([i, cols, rows, col_point, row_point, tables])
                # print(cols, rows)

    except Exception as e:
        print(e)
    # for index, i in enumerate(cells[0][5]):
    #     Image.fromarray(i[0]).show()
    # print(55555555, len(cells))
    if cells:
        return cells
    else:
        return []
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


def generate_table(cell, table_data):
    pos, cols, rows, col_point, row_point, tables = cell[0][1], cell[1], cell[2], cell[3], cell[4], cell[5]
    # print(pos)
    col_point = sorted(col_point)
    row_point = sorted(row_point)
    tables = sorted(tables, key=lambda i: i[1][3])[:-1]
    tables = sorted(tables, key=lambda i: i[1][0] + i[1][1])
    for i in tables:
        d = {'col_begin': 0, 'col_end': 0, 'row_begin': 0, 'row_end': 0}
        for index, value in enumerate(col_point):
            if index == 0:
                d_range = 1
            else:
                d_range = 1
            if i[1][0] > col_point[index] - d_range:
                d['col_begin'] = index
        for index, value in enumerate(col_point):
            if index == len(col_point) - 1:
                d_range = 1
            else:
                d_range = 1
            if i[1][0] + i[1][2] < col_point[index] + d_range:
                d['col_end'] = index
                break
        for index, value in enumerate(row_point):
            if index == 0:
                d_range = 1
            else:
                d_range = 1
            if i[1][1] > row_point[index] - d_range:
                d['row_begin'] = index
        for index, value in enumerate(row_point):
            if index == len(row_point) - 1:
                d_range = 1
            else:
                d_range = 1
            if i[1][1] + i[1][3] < row_point[index] + d_range:
                d['row_end'] = index
                break
        i.append(d)
    for i in tables:
        try:
            texts = []
            x0, y0, x1, y1 = pos[0]+i[1][0], pos[1]+i[1][1], pos[0]+i[1][0]+i[1][2], pos[1]+i[1][1]+i[1][3]
            for td in table_data:
                central_point = (0.5*(td[1][0]+td[1][2]), 0.5*(td[1][1]+td[1][3]))
                if x0 < central_point[0] < x1 and y0 < central_point[1] < y1:
                    texts.append(td)

            texts = sorted(texts, key=lambda x: x[1][1])
            new_texts = []
            for t_index, t in enumerate(texts):
                if t_index == 0:
                    new_texts.append([t])
                    continue
                if new_texts[-1][-1][-1][3] < 0.5*(t[1][1]+t[1][3]) < new_texts[-1][-1][-1][1]:
                    new_texts[-1].append(t)
                else:
                    new_texts.append([t])
            # print(111111111111, new_texts[0][0])
            # print(new_texts
            new_texts = [sorted(n, key=lambda x:x[1][0]) for n in new_texts]
            texts = sum(new_texts, [])
            texts = ''.join([i[0] for i in texts])
            # print(texts)
            i.append([texts, i[1][2], i[1][3]])
        except Exception as e:
            print(e)
    new_table = []
    for i in tables:
        new_table.append([i[2], i[3]])

    return new_table, rows, cols, pos


if __name__ == '__main__':

    im_white = np.zeros((842, 595, 3), np.uint8)
    im_white = 255 - im_white

    img_name = Image.open('a.jpg')
    img_name.thumbnail((595.5, 842.25), Image.ANTIALIAS)
    # img_name = img_name.convert('RGB')
    img = np.array(img_name)
    tables = extract_table(img)
    table = tables[0]
    print(table)
    # import pickle
    # text_data = pickle.load(open('data.pkl', 'rb'))
    # if tables:
    #     for table in tables:
    #         print(generate_table(table, text_data=text_data))
    # Image.fromarray(img).show()
    cv2.rectangle(img, (26, 77), (26+544, 77+693), (0, 0, 0), 3)
    Image.fromarray(img).show()
