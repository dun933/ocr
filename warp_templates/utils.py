def tem_match(origin_test, origin_template):

    test = []
    for i in origin_test:
        tem_test = []
        for index, j in enumerate(i[1]):
            j.insert(0, i[0][index])
            tem_test.append(j)
        test.append([i[0], tem_test])

    tem_template = {}
    for i in origin_template:
        tem_template[i[0]] = i[1]
    template = {}
    for k, v in tem_template.items():
        template[k] = [[i, j] for i, j in zip(k, v)]

    set_list = []
    for i, _ in enumerate(test):
        set_list.append(set(test[i][0]))
    # print(set_list)
    result_list = []
    for k, v in template.items():
        key_set = set(k)
        # print(k)

        num_list, k_str = [], []
        k_str.append(k)
        # print(k)
        for set_test in set_list:
            num_list.append(len(key_set.intersection(set_test)))

        find = max(num_list)

        if find:
            find_index = [i for i, v in enumerate(num_list) if v == find]
        else:
            find_index = []

        match_rate = 0
        match_index = -1
        # match_word = []
        for i in find_index:
            # print('TEST:', test[i][0])
            # print('TEMPLATE KEY:', k)
            if test[i][0][0] == k[0] and find/len(k) > match_rate:
                match_index = i
                match_rate = find/len(k)

        # print('find:', find, 'match rate:', match_rate)
        if match_index != -1 and match_rate > 0.7 and len(k) > 1:
            # print(test[match_index][1])
            uniq_check = []
            for i in test[match_index][1]:
                # print('i:', i,'v:', v)

                if i[0] in k and i[0]:
                    uniq_check.append(i[0])

                if len(uniq_check)==len(set(uniq_check)):
                    k_list = []
                    for j in v:
                        # print(j)
                        if i[0] == j[0]:
                            tem_cor = j[1:]
                            break
                        else:
                            tem_cor = []

                    if len(tem_cor) > 0 :
                        k_list.append(i[0])
                        k_list.append(i[1:])
                        k_list.append(tem_cor[0])
                        k_str.append(k_list)
                        # break

            if len(k_str) > 1:
                result_list.append(k_str)
    return result_list

