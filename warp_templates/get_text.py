from PIL import Image
import os
from collections import defaultdict

# from ctpn.ctpn_blstm_test import text_predict
# from densent_ocr.model import predict
# from Rec_text import rec_txt
# from pan.predict import text_predict
# from crnn_torch.model1 import predict
from xml.etree import ElementTree as ET

from full_ocr_local import single_ocr
from viterbi import calculate


def get_texts(xml, found, img_path, task_id):
    dom = ET.parse(xml)
    obj = dom.findall("./object")
    dic = {}
    texts = defaultdict(str)
    ori_img = np.array(found)
    _, ori = cv2.imencode('.jpg', ori_img)
    ori = base64.b64encode(ori.tostring())
    for ob in obj:
        name = str(ob.getchildren()[0].text)
        if 'extract' in name:
            bnd_box = ob.findall("bndbox")[0]
            x_min = bnd_box.findall("xmin")[0].text
            y_min = bnd_box.findall("ymin")[0].text
            x_max = bnd_box.findall("xmax")[0].text
            y_max = bnd_box.findall("ymax")[0].text
            dic[name] = [int(x_min), int(y_min), int(x_max), int(y_max)]
    _, img = cv2.imencode('.jpg', found)
    img = base64.b64encode(img.tostring())
    data = {'img': img, 'scale_w': scale_w, 'scale_h': scale_h, 'ori_img': ori}
    images = requests.post('http://172.30.20.154:32021/text_predict', data=data)
    for key, value in dic.items():
        if key == '商标-extract':
            save_path = '/' + '/'.join(img_path.split('/')[1:5]) + '/trademark/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = save_path + str(task_id) + '.jpg'
            Image.fromarray(found[value[1]: value[3], value[0]: value[2]]).resize((500, 300)).save(save_path)
            texts[key.replace('-extract', '')] = save_path
            continue
        try:
            # if images:
            #     image_positions = [[i[0], rec_txt(i[1])]
            #                        for i
            #                        in
            #                        images]
            new_images = []
            for i in images:
                if i[0][1] > value[1] and i[0][7] < value[3]:
                    new_images.append(i)
            new = []
            for i in new_images:
                if i[0][0] > value[0] and i[0][6] < value[2]:
                    new.append(i)
                elif i[0][0] < value[0] and (value[0] < i[0][6] < value[2]):
                    i[0][0] = value[0]
                    new.append([i[0], found[i[0][1]: i[0][7], value[0]:i[0][6]]])
                elif (value[2] > i[0][0] > value[0]) and i[0][6] > value[2]:
                    i[0][6] = value[2]
                    new.append([i[0], found[i[0][1]: i[0][7], i[0][0]:value[2]]])
                elif i[0][0] < value[0] and i[0][6] > value[2]:
                    i[0][0] = value[0]
                    i[0][6] = value[2]
                    new.append([i[0], found[i[0][1]: i[0][7], value[0]:value[2]]])
            if new:
                image_positions = []
                for j in new:
                    if j[1].any():
                        _, img = cv2.imencode('.jpg', j[1])
                        img = base64.b64encode(img.tostring())
                        data = {'img': img}
                        content = requests.post('http://172.30.20.154:32020/predict', data=data).json()[:2]
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
                        image_positions.append([j[0], content])
                texts[key.replace('-extract', '')] = single_ocr(image_positions).replace('\n', '')
            else:
                _, img = cv2.imencode('.jpg', found)
                img = base64.b64encode(img.tostring())
                data = {'img': img}
                content = requests.post('http://172.30.20.154:32020/predict', data=data).json()[:2]
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
                texts[key.replace('-extract', '')] = content
        except Exception as e:
            print(e)
            continue
    if '企业名称' in texts.keys():
        texts['企业名称'] = texts['企业名称'].replace('企业名称', '')
    return texts
