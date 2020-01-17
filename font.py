import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import cv2
import numpy as np
import os
from PIL import Image
from font_model import model
import pickle

label_dict = {0: 'noise', 1: 'word', 2: 'table', 3: 'image'}
# label_dict = pickle.load(open('cls.pkl', 'rb'))
print(label_dict)
model = model()
model.load_weights(r'cls.hdf5')


def predict_img(img):
    img = np.array(img.convert('RGB').resize((32, 32)))
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img /= 255
    label = model.predict(img)
    # print(label)
    label = label_dict[np.argmax(label[0])]
    return label


# processing letter by letter boxing
def process_letter(thresh, output):
    # assign the kernel size
    kernel = np.ones((2, 1), np.uint8)  # vertical
    # use closing morph operation then erode to narrow the image
    temp_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    # temp_img = cv2.erode(thresh,kernel,iterations=2)
    letter_img = cv2.erode(temp_img, kernel, iterations=1)

    # find contours
    contours, _ = cv2.findContours(letter_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop in all the contour areas
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x - 1, y - 5), (x + w, y + h), (0, 255, 0), 1)

    return output


# processing letter by letter boxing
def process_word(thresh, output):
    # assign 2 rectangle kernel size 1 vertical and the other will be horizontal
    kernel = np.ones((2, 1), np.uint8)
    kernel2 = np.ones((1, 4), np.uint8)
    # use closing morph operation but fewer iterations than the letter then erode to narrow the image
    temp_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # temp_img = cv2.erode(thresh,kernel,iterations=2)
    word_img = cv2.dilate(temp_img, kernel2, iterations=1)

    (contours, _) = cv2.findContours(word_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x - 1, y - 5), (x + w, y + h), (0, 255, 0), 1)

    return output


# processing line by line boxing
def process_line(thresh, output, image):
    # assign a rectangle kernel size	1 vertical and the other will be horizontal
    kernel = np.ones((3, 9), np.uint8)
    kernel2 = np.ones((1, 4), np.uint8)

    # use closing morph operation but fewer iterations than the letter then erode to narrow the image
    temp_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2, iterations=2)
    line_img = cv2.dilate(temp_img, kernel, iterations=5)
    # i = str(np.random.rand())
    # cv2.imwrite('%s/%s.jpg' % ('line/', i), line_img)
    contours, _ = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_y_list = []
    img_list = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        crop = image.crop((x, y, x+w, y+h))
        if predict_img(crop) == 'word':
            # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x_y_list.append([x, y+2, x + w, y + h])
            # crop.save('word/' + str(np.random.randint(999999)) + '.jpg')
        if predict_img(crop) == 'image':
            # cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # crop.save('image/' + str(np.random.randint(999999)) + '.jpg')
            if w / h < 7:
                img_list.append([x, y + 2, x + w, y + h])
                # cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # if predict_img(crop) == 'table':
        #     cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     # crop.save('table/' + str(np.random.randint(999999)) + '.jpg')
    # Image.fromarray(output).show()
    if img_list:
        img_list = sorted(img_list, key=lambda x: x[1])
    # print('111111111111', img_list)
    return [x_y_list, img_list]  # output, thresh


# processing par by par boxing
def process_par(thresh, output):
    # assign a rectangle kernel size
    kernel = np.ones((5, 5), 'uint8')
    par_img = cv2.dilate(thresh, kernel, iterations=3)

    (contours, _) = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return output


# processing margin with paragraph boxing
def process_margin(thresh, output):
    # assign a rectangle kernel size
    kernel = np.ones((20, 5), 'uint8')
    margin_img = cv2.dilate(thresh, kernel, iterations=5)

    (contours, _) = cv2.findContours(margin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return output


def denoise(thresh):

    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((1, 2), np.uint8)
    temp_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2, iterations=1)
    line_img = cv2.dilate(temp_img, kernel, iterations=3)

    contours, _ = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    array_h = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        array_h.append(h)
    h_mean = np.mean(array_h)
    if len(contours) > 20:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < (h_mean / 3) or (h * w) < 800:
                thresh[y-5:y+h, x-2:x+w] = 0
    else:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (h * w) < 800:
                thresh[y-5:y+h, x-2:x+w] = 0
    return thresh


def get_detail(img):
    try:
        im = np.array(img)

        output_line = im.copy()
        try:
            blur = cv2.GaussianBlur(im, (5, 5), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

        except Exception as e:
            print('Stage 1:', file, e)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        denoised = denoise(th)
        # denoise_img = Image.fromarray(denoised)

        # output_letter = process_letter(denoised, output_letter)
        # letter_img = Image.fromarray(output_letter)
        # letter_img.save('letter' + '//' + file)
        bb = process_line(denoised, output_line, img)
        # # output_par = process_par(binary, output_line)
        # final = Image.fromarray(output_line)
        return bb
    except Exception as ex:
        print('font_error', ex)
        return [[], []]


#
# if __name__ == '__main__':
#
#     root = r'F:\paragraph_restore\imgs'
#     file = '14 建设工程档案报送责任书1.jpg'
#
#     image = Image.open(root + '//' + file)
#     img = np.array(Image.open(root + '//' + file))
#
#     print(get_detail(img))
