import cv2
import math
import numpy as np
from PIL import Image, ImageEnhance

import torch
import base64
import requests
from viterbi import calculate
from config_url import DETECT_URL, RECOGNISE_URL

# DETECT_URL = 'http://172.30.81.191:32021/text_predict'
# RECOGNISE_URL = 'http://172.30.81.191:32020/predict'


def remove_line(src, FLAG, call_limit=10, gap_rate=0.05, epsilon=1e-6):
    if call_limit:
        if FLAG == 0:
            contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            horizontal_coor = []
            for i in range(len(contours)):
                x, y, w, h = cv2.boundingRect(np.array(contours[i]))
                horizontal_coor.append((x, y , w, h))

            if len(horizontal_coor) >= 2:
                horizontal_coor = sorted(horizontal_coor, key=lambda i: i[0])
                if abs(horizontal_coor[1][0] - horizontal_coor[0][0]) / (horizontal_coor[1][0] + epsilon) > gap_rate:
                    x, y, w, h = horizontal_coor[0]
                    src[y:y+h, x:x+w] = 0

                elif abs(horizontal_coor[-1][0] - horizontal_coor[-2][0]) / (horizontal_coor[-1][0] + epsilon) > gap_rate:
                    x, y, w, h = horizontal_coor[-1]
                    src[y:y+h, x:x+w] = 0
                else:
                    return src
            else:
                return src
            call_limit = call_limit - 1
            return remove_line(src, FLAG=0, call_limit=call_limit)

        elif FLAG == 1:
            contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vertical_coor = []
            for i in range(len(contours)):
                x, y, w, h = cv2.boundingRect(np.array(contours[i]))
                vertical_coor.append((x, y, w, h))

            if len(vertical_coor) >= 2:
                vertical_coor = sorted(vertical_coor, key=lambda i: i[1])
                if abs(vertical_coor[1][1] - vertical_coor[0][1]) / (vertical_coor[1][1] + epsilon) > gap_rate:
                    x, y, w, h = vertical_coor[0]
                    src[y:y+h, x:x+w] = 0

                elif abs(vertical_coor[-1][1] - vertical_coor[-2][1]) / (vertical_coor[-1][1] + epsilon) > gap_rate:
                    x, y, w, h = vertical_coor[-1]
                    src[y:y+h, x:x+w] = 0
                else:
                    return src
            else:
                return src
            call_limit = call_limit - 1
            return remove_line(src, FLAG=1, call_limit=call_limit)
    else:
        return src


def table_lines(src, coor, origin, file=None, horizontal=[], vertical=[], scale=30):
    roi = origin[coor[1]:coor[1]+coor[3],coor[0]:coor[0]+coor[2],:]

    src_height, src_width = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # print(src_height)

    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    roi_horizontal = thresh
    roi_vertical = thresh

    roi_horizontalsize = int(src_height / scale)
    roi_horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (roi_horizontalsize, 1))
    roi_horizontal = cv2.erode(roi_horizontal, roi_horizontalStructure, iterations=1)
    roi_horizontal = cv2.dilate(roi_horizontal, roi_horizontalStructure, iterations=1)
    roi_horizontal = cv2.blur(roi_horizontal, (5, 5))

    roi_verticalsize = int(src_width / scale)
    roi_verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, roi_verticalsize))
    roi_vertical = cv2.erode(roi_vertical, roi_verticalStructure, (-1, -1))
    roi_vertical = cv2.dilate(roi_vertical, roi_verticalStructure, (-1, -1))
    roi_vertical = cv2.blur(roi_vertical, (5, 5))
    # Image.fromarray(vertical + horizontal + roi_horizontal + roi_vertical) .show()


    # 霍夫直线变换,检测倾斜角度，进行旋转
    edges = cv2.Canny(vertical, 30, 240)
    lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 200, minLineLength=20, maxLineGap=100)

    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        lengths.append([x1, y1, x2, y2, length])
        # print(line, length)
        # cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制所有直线

    # 绘制最长的直线
    lengths.sort(key=lambda i:i[-1])
    longest_line = lengths[-1]
    print(longest_line)
    x1, y1, x2, y2, length = longest_line
    # cv2.line(src, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 绘制直线

    # 计算这条直线的旋转角度
    angle = math.acos((x2 - x1) / length)
    # print(angle)  # 弧度形式
    degree = angle * (180 / math.pi)
    # print(degree)  # 角度形式

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
    # oints = cv2.bitwise_and(horizontal, vertical)
    # wi = cv2.findNonZero(oints)
    #
    # xx = [i[0][0] for i in wi]
    # yy = [i[0][1] for i in wi]
    # if min(xx) > 0.05*w:
    cv2.line(vertical, (5, 5), (5, h - 5), (255, 255, 255), 3)
    # if max(xx) < 0.95*w:
    cv2.line(vertical, (w - 5, y + 5), (x + w - 5, y + h - 5), (255, 255, 255), 3)
    # if min(yy) > 0.05*h:
    cv2.line(horizontal, (5, 5), (w - 5, 5), (255, 255, 255), 3)
    # if max(yy) < 0.95*h:
    cv2.line(horizontal, (5, h - 5), (w - 5, h - 5), (255, 255, 255), 3)
    
    origin_mask = roi_horizontal+roi_vertical
    horizontal = horizontal + roi_horizontal
    vertical = vertical + roi_vertical
    mask = horizontal + vertical
    

    # erode_size = 3
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size), (-1, -1))
    # mask = cv2.morphologyEx(~mask, cv2.MORPH_OPEN, element)
    # mask = ~mask
    # mask = cv2.blur(mask, (5, 5))
    # x, y, w, h = cv2.boundingRect(mask)
    # Image.fromarray(mask).show()
    # cv2.rectangle(mask, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 1)

    # print(horizontal.shape, vertical.shape)
    joints = cv2.bitwise_and(horizontal, vertical)
    joints = cv2.dilate(joints, None)
    # Image.fromarray(joints).show()


    if not joints.any():
        return False

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(hierarchy)
    contours_poly = [''] * len(contours)
    boundRect = [''] * len(contours)
    rois = []
    for i, j in zip(range(len(contours)), hierarchy[0]):
        # print(j)
        if j[3]==0:
            area = cv2.contourArea(contours[i])
            if area < 100:
                continue

            contours_poly[i] = cv2.approxPolyDP(np.array(contours[i]), 3, True)

            # cv2.drawContours(src, contours_poly[i], 0, (0, 255, 0), thickness=3)
            boundRect[i] = x, y ,w ,h = cv2.boundingRect(np.array(contours_poly[i]))
            # cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 5)

            roi = joints[boundRect[i][1]:boundRect[i][1] + boundRect[i][3],
                  boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]

            joints_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(joints_contours)
            # if len(joints_contours) < 1:
            #     print(111)
            #     continue

            # ytt = src[boundRect[i][1]:boundRect[i][1] + boundRect[i][3], boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]
            # scale = src.shape[0]/ori_image.shape[0]
            ytt = src[boundRect[i][1]:boundRect[i][1] + boundRect[i][3], boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]
            src[boundRect[i][1]:boundRect[i][1] + boundRect[i][3], boundRect[i][0]:boundRect[i][0] + boundRect[i][2]] = 0

            # rois.append(src(boundRect[i]).clone())
            rois.append([ytt, list(boundRect[i])])
        elif j[3] > 0:
            contours_poly[i] = cv2.approxPolyDP(np.array(contours[i]), 3, True)
            boundRect[i] = x, y, w, h = cv2.boundingRect(np.array(contours_poly[i]))
            # cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            pass


    # Image.fromarray(src).show()
    # cv2.imwrite('mask/' + file, src)
    new_con, _ = cv2.findContours(joints, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Image.fromarray(joints).show()
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
        if abs(num_x[index] - num_x[index + 1]) < 12:
            num_x[index + 1] = num_x[index]
            xs.add(num_x[index])
    ys = set()
    for index in range(len(num_y) - 1):
        if abs(num_y[index] - num_y[index + 1]) < 12:
            num_y[index + 1] = num_y[index]
            ys.add(num_y[index])
    print(xs, ys)
    # print(rois)
    return len(xs) - 1, len(ys) - 1, list(xs), list(ys), rois


def draw_line(src, scale=30):
    if not src.data:
        print('not picture')

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Image.fromarray(gray).show()
    erode_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size), (-1, -1))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, element)

    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -10)
    # Image.fromarray(thresh).show()
    horizontal = thresh
    vertical = thresh

    horizontalsize = int(horizontal.shape[0] / scale)

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.blur(horizontal, (5, 5))

    verticalsize = int(vertical.shape[1] / scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    vertical = cv2.blur(vertical, (5, 5))
    # Image.fromarray(horizontal+vertical).show()

    # 没有交点则空
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
    if len(xs) <= 1:
        return [], []


    horizontal = remove_line(horizontal, FLAG=0, gap_rate=0.05)
    vertical = remove_line(vertical, FLAG=1, gap_rate=0.05)

    mask = horizontal + vertical
    # Image.fromarray(mask).show()
    # cv2.imwrite('mask/' + file, mask)
    joints = cv2.bitwise_and(horizontal, vertical)
    # Image.fromarray(joints).show()
    

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [''] * len(contours)
    boundRect = [''] * len(contours)
    rois = []

    # 原本为检测多个表格，现在输入固定只有一个表格，所以舍弃
    # for i in range(len(contours)):
    #     area = cv2.contourArea(contours[i])
    #     if area < 100:
    #         continue
    #     # print(contours[i])
    #     contours_poly[i] = cv2.approxPolyDP(contours[i], 3, True)
    #     # print(contours_poly[i])
    #     boundRect[i] = cv2.boundingRect(np.array(contours_poly[i]))

    #     roi = joints[boundRect[i][1]:boundRect[i][1] + boundRect[i][3],
    #           boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]

    #     joints_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(joints_contours) < 1:
    #         continue

    #     ytt = src[boundRect[i][1]:boundRect[i][1] + boundRect[i][3], boundRect[i][0]:boundRect[i][0] + boundRect[i][2]]
    #     # Image.fromarray(ytt).show()
    #     rois.append([ytt, list(boundRect[i])])
    rois.append([src, [0, 0, src.shape[1], src.shape[0]]])
    return rois, src, horizontal, vertical


def extract_table(image, table_image):
    image = 255 - image
    # image = Image.fromarray(ori_image).copy()
    # image.thumbnail((600, 600), Image.ANTIALIAS)
    image = image.copy()
    # print(image.shape)

    rois, src, horizontal, vertical = draw_line(image)
    print(7777777777, image.shape, len(rois))
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
    #     print(i[1])
    cells = []
    # print(tables)
    for i in tables:
        # print(i)
        cols, rows, col_point, row_point, tables = table_lines(i[0], i[1], table_image, horizontal=horizontal[i[1][1]:i[1][1]+
                                                               i[1][3],i[1][0]:i[1][0]+i[1][2]],\
                                                               vertical=vertical[i[1][1]:i[1][1]+i[1][3],i[1][0]:i[1][0]+i[1][2]])
        print('cols and rows', cols, rows)
        # Image.fromarray(i[0]).show()
        if cols > 1 and rows > 1:
            cells.append([i, cols, rows, col_point, row_point, tables])

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
    # print(11111111111111, pos)
    table_im = ori_img[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]

    # table_line_regions = text_predict(table_im, 1, 1, table_im)
    _, d = cv2.imencode('.jpg', table_im)
    d = base64.b64encode(d)
    d = d.decode()
    data = {'img': d, 'scale_w': 1, 'scale_h': 1, 'ori_img': d}
    table_line_regions = requests.post(DETECT_URL, data=data).json()
    table_line_regions = [[xx[0], cv2.imdecode(np.fromstring(base64.b64decode(xx[1]), np.uint8), cv2.IMREAD_COLOR)] for xx in table_line_regions]

    word_list = []
    for region in table_line_regions:
        region_y = [region[0][1], region[0][5]]
        region_x = [region[0][0], region[0][2]]
        
        # content = predict(Image.fromarray(region[1]).convert('L'))
        # content = (content[0][0], content[0][1], content[1])
        # Image.fromarray(region[1]).show()
        _, tmp = cv2.imencode('.jpg', region[1])
        tmp = base64.b64encode(tmp)
        tmp = tmp.decode()
        data = {'img': tmp}
        content = requests.post(RECOGNISE_URL, data=data).json()
        # print(content)
        
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
    # print(word_list)
    for region in table_line_regions:
        cv2.rectangle(table_im, (region[0][0], region[0][1]), (region[0][6], region[0][7]), (0, 255, 0), 1)
    for i in word_list:
        cv2.rectangle(table_im, (i[0][0], i[0][1]), (i[0][6], i[0][7]), (255, 0, 0), 1)
    
    Image.fromarray(table_im).save('single_word.jpg')

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

    for i in tables:
        cv2.rectangle(table_im, (i[1][0], i[1][1]), (i[1][0] + i[1][2], i[1][1] + i[1][3]), (255, 0, 0), 1)
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
    Image.fromarray(table_im).save('single_word.jpg')
    new_table = []
    for i in tables:
        new_table.append([i[2], i[3]])

    return new_table, rows, cols, pos


if __name__ == '__main__':
    import fitz
    import os
    for file in os.listdir('predict'):
        # if not file.endswith('docx') and not any(word in file for word in ['mask', 'predict']):
        if file == '1 20131225 高新风投 评估报告_95.jpg':
            img_path = r'predict/{}'.format(file)
            img = Image.open(img_path).convert('RGB')
            # img = ImageEnhance.Sharpness(img).enhance(5)
            img = ImageEnhance.Color(img).enhance(3)
            img.show('origin')

            # img.show()
            # img_name = img_name.convert('RGB')
            print(file)
            img = np.array(img)

            try:
                a = extract_table(img, file)
                im_white = 255 - np.zeros_like(img)
                i = ['table', generate_table(a[0], img)]

                print(i)
                for coor in i[1][0]:
                    im_white[coor[1][2], coor[1][1], :] = 0
                Image.fromarray(im_white).show()
                i = [i, 'table', 10.5, 1, 0]
                i = [i[0][1], i[1], i[2], i[3], i[4]]

                from restore_table import restore_table
                from docx import Document
                doc = Document()
                doc = restore_table(doc, i, Image.fromarray(img))
                doc.save('word/' + file.replace('.jpg', '.docx'))
            except Exception as e:
                print(e)



