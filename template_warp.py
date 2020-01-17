import pickle

import cv2
from PIL import Image
import numpy as np

# from densent_ocr.model import predict
# from Rec_text import rec_txt
# from crnn_torch.model import predict1
from warp_templates.utils import tem_match
from WordSegmentation import wordSegmentation


def PolygonSort(corners):
    n = len(corners)
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    cornersWithAngles = []
    for x, y in corners:
        an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        cornersWithAngles.append((x, y, an))
    cornersWithAngles.sort(key=lambda tup: tup[2])

    def function1(x, y, an):
        return x, y

    corners_sorted = map(function1, cornersWithAngles)
    n = len(corners_sorted)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners_sorted[i][0] * corners_sorted[j][1]
        area -= corners_sorted[j][0] * corners_sorted[i][1]
    area = abs(area) / 2.0
    return area


def PolygonArea(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def get_pick(ori_img, position, key, scale, max_len, max_content):
    tt_img = ori_img[position[0][1]:position[0][7], position[0][0]:position[0][6]]
    Image.fromarray(tt_img).save('tt.jpg')
    tt_img = cv2.imread('tt.jpg')
    img = cv2.resize(tt_img, (int(tt_img.shape[1] * scale), int(tt_img.shape[0] * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = wordSegmentation(gray, kernelSize=25, sigma=11, theta=7, minArea=100)
    pick = []
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        (x, y, w, h) = wordBox
        pick.append([x, y, x+w, y+h])
    # mser = cv2.MSER_create()
    # regions = mser.detectRegions(gray)
    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    # cv2.polylines(img, hulls, 1, (0, 255, 0))
    # keep = []
    # for c in hulls:
    #     x, y, w, h = cv2.boundingRect(c)
    #     keep.append([x, y, x + w, y + h])
    # keep2 = np.array(keep)
    # pick = nms.non_max_suppression_fast(keep2, 0.5)
    contents = []
    for (startX, startY, endX, endY) in pick:
        word = tt_img[int(startY / scale):int(endY / scale), int(startX / scale):int(endX / scale)]
        try:
            if word.any() and int(word.shape[1] / (word.shape[0] * 1.0 / 32)) > 8:
                # content = rec_txt(word)
                content = predict(Image.fromarray(word).convert('L'))
                # for index, i in enumerate(content[1]):
                #     # if i[0] > 0.9:
                #         content[0][index] = content[0][index][0]
                #         content[1].pop(index)
                # content = calculate(content)
                contents.append(content)
            else:
                continue
        except Exception as e:
            continue
    content_num = 0
    for i in contents:
        if i in key and (i != key or len(key) == 1) and i != '' and len(i) == 1:
            content_num += 1
    if content_num > max_len:
        max_len = content_num
        max_content = pick, scale, tt_img
    if scale == 5 and max_content:
        return max_content
    if content_num >= len(key) / 2 or scale > 15:

        # print(pick, scale, tt_img)
        return pick, scale, tt_img
    else:
        return get_pick(ori_img, position, key, scale + 1, max_len, max_content)

# def get_pick(ori_img, position, scale):
#     key = position[1]
#     tt_img = ori_img[position[0][1]:position[0][7], position[0][0]:position[0][6]]
#     Image.fromarray(tt_img).save('tt.jpg')
#     img = cv2.resize(tt_img, (int(tt_img.shape[1] * scale), int(tt_img.shape[0] * scale)))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     mser = cv2.MSER_create()
#     regions = mser.detectRegions(gray)
#     hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
#     cv2.polylines(img, hulls, 1, (0, 255, 0))
#     keep = []
#     for c in hulls:
#         x, y, w, h = cv2.boundingRect(c)
#         keep.append([x, y, x + w, y + h])
#     keep2 = np.array(keep)
#     pick = nms.non_max_suppression_fast(keep2, 0.5)
#     contents = []
#     for (startX, startY, endX, endY) in pick:
#         word = tt_img[int(startY / scale):int(endY / scale), int(startX / scale):int(endX / scale)]
#         try:
#             content = predict(Image.fromarray(word).convert('L')).replace('：', '')
#             contents.append(content)
#         except Exception as e:
#             continue
#     content_num = 0
#     for i in contents:
#         if i in key and i != key:
#             content_num += 1
#     if content_num >= 1 or scale > 15:
#         # print(pick, scale, tt_img)
#         return pick, scale, tt_img
#     else:
#         # print(1111)
#         return get_pick(ori_img, position, scale + 1)


def get_image_position(ori_img, matches):
    image_positions = []
    for match in matches:
        scale = 1
        max_len = 0
        max_content = []
        pick, scale, tt_img = get_pick(ori_img, match[0], match[1], scale, max_len, max_content)
        words = []
        for (startX, startY, endX, endY) in pick:
            word = tt_img[int(startY / scale):int(endY / scale), int(startX / scale):int(endX / scale)]
            try:
                if word.any() and int(word.shape[1] / (word.shape[0] * 1.0 / 32)) > 8:
                    content = predict(Image.fromarray(word).convert('L'))
                    for index, i in enumerate(content[1]):
                        if i[0] > 0.9:
                            content[0][index] = content[0][index][0]
                            content[1].pop(index)
                    content = calculate(content)
                else:
                    continue
            except Exception as e:
                continue
            words.append(
                [content, match[0][0][0] + int(startX / scale), match[0][0][1] + int(startY / scale),
                 match[0][0][0] + int(endX / scale),
                 match[0][0][1] + int(endY / scale)])
        image_positions.append([words, match[2]])
    return image_positions


def cmp_items(a, b):
    corners_a = []
    for i in a:
        corners_a.append((i[1][1][0], i[1][1][1]))
    corners_b = []
    for i in b:
        corners_b.append((i[1][1][0], i[1][1][1]))
    area_a = PolygonArea(corners_a)
    area_b = PolygonArea(corners_b)
    if area_a > area_b:
        return 1
    elif area_a == area_b:
        return 0
    else:
        return -1


def get_match_point(image, image_positions, FT):
    template_positions = pickle.load(open('warp_templates/{}/template.pkl'.format(FT), 'rb'))
    # image = Image.open(image).convert('RGB')
    # # image = Image.fromarray(image)
    # image.thumbnail((2000, 2000), Image.ANTIALIAS)
    # image = np.array(image)
    # print(1111111, image.shape)
    # images = text_predict(image)
    # image_positions = [[i[0], predict(Image.fromarray(i[1]).convert('L')).replace('“', '').replace('‘', '')] for i in
    #                    images]
    # pickle.dump(image_positions, open('test.pkl', 'wb'))
    result = tem_match(image_positions, template_positions)
    # match_positions = get_image_position(image, match)
    # result = word_match(match_positions)
    # results = list(itertools.combinations(result, 4))
    src_pts = []
    dst_pts = []
    print(2222222222222222, result)
    for i in result:
        for j in i[1:]:
            src_pts.append([j[1][0], j[1][1]])
            dst_pts.append([j[2][0], j[2][1]])
    # print(33333333333333, src_pts)
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    # results.sort(key=cmp_to_key(cmp_items))
    # result = results[-1]
    # src_pts = np.float32([[m[1][1][0], m[1][1][1]] for m in result])
    # dst_pts = np.float32([[m[1][2][0], m[1][2][1]] for m in result])
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(M)
    return M


def get_warp(image, image_positions, FT):
    template = cv2.imread('warp_templates/{}/template.jpg'.format(FT))
    M = get_match_point(image, image_positions, FT)
    h, w = template.shape[:2]
    # pts1 = np.float32([[point[0][1][1][0], point[0][1][1][1]], [point[1][1][1][0], point[1][1][1][1]],
    #                    [point[2][1][1][0], point[2][1][1][1]], [point[3][1][1][0], point[3][1][1][1]]])
    # pts2 = np.float32(
    #     [[point[0][1][2][0], point[0][1][2][1]], [point[1][1][2][0], point[1][1][2][1]], [point[2][1][2][0], point[2][1][2][1]],
    #      [point[3][1][2][0], point[3][1][2][1]]])
    # print(pts1, pts2)
    # M = cv2.getPerspectiveTransform(np.float32([[0, 0], [0, h_o], [w_o, h_o], [w_o, 0]]), np.float32([[0, 0], [0, h], [w, h], [w, 0]]))
    # found = cv2.warpPerspective(image, M, (w, h))
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    # dst = cv2.perspectiveTransform(pts, M)
    # M = cv2.getPerspectiveTransform(np.float32(dst), np.float32(pts))
    found = cv2.warpPerspective(image, M, (w, h))
    Image.fromarray(found).save('test.jpg')
    return found

