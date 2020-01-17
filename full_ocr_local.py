def single_ocr(results):
    document = ''
    results = sorted(results, key=lambda i: i[0][1])
    new_results = results
    line_images = []
    cut_index = 0
    curr_index = 0
    for index, i in enumerate(new_results):
        if index == len(new_results) - 1:
            if cut_index < index:
                line_images.append(new_results[cut_index:index])
                line_images.append(new_results[index:])
            else:
                line_images.append(new_results[index:])
            break
        # if abs(new_results[index + 1][0][1] - new_results[index][0][1]) > (
        #         new_results[index][0][7] - new_results[index][0][1]) * 4 / 5:
        #     line_images.append(new_results[cut_index: index + 1])
        #     cut_index = index + 1
        if abs(new_results[index + 1][0][1] - new_results[curr_index][0][1]) > (
                new_results[curr_index][0][7] - new_results[curr_index][0][1]) * 4 / 5:
            line_images.append(new_results[cut_index: index + 1])
            cut_index = index + 1
            curr_index = index + 1
    for index, i in enumerate(line_images):
        line_images[index] = sorted(i, key=lambda a: a[0][0])
    texts = []
    for i in line_images:
        text = ''
        for index, j in enumerate(i):
            try:
                if index == len(i) - 1:
                    text += j[1]
                elif abs(i[index + 1][0][0] - i[index][0][6]) > 3 * (abs(i[index][0][6] - i[index][0][0]) / len(i[index][1])):
                    text += j[1] + ' '
                else:
                    text += j[1]
            except:
                continue
        texts.append([i[0][0], text])
    for i in texts:
        document += i[1] + '\n'
    # with open('/home/ddwork/projects/ocr_result/' + ''.join(pdf_files.split('/')[:-1]) + os.path.split(pdf_files)[1].split('.')[0] + '.txt', 'w', encoding='utf-8') as f:
    #     for i in document.split('\n'):
    #         f.write(i + '\n')
    return document
