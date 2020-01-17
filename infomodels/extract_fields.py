import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import re
import pickle
import base64

# from bert_ner.bert_predict import predict_ner



def extract_mode(text, modes, filter_char):
    value = ''
    for mode in modes:
        for line in text.split('\n'):
            v = re.compile(mode).search(''.join([i for i in line if i not in filter_char]))
            if v:
                value = v.groups()[0]
                if len(value) > 1:
                    return value
    return value


def extract_mode_full(text, modes, filter_char):
    """
     提取软件名称
    :return: str（软件名称）
    """
    value = ''
    text = ''.join([i for i in text.replace('\n', '') if i not in filter_char])
    for mode in modes:
        se = re.compile(mode).search(text)
        if se:
            value = se.groups()[0]
            return value
    return value


def extract_mode_full_1(text, modes, filter_char):
    """
     提取软件名称
    :return: str（软件名称）
    """
    value = ''
    # text = ''.join([i for i in text if i not in filter_char])
    # print(text)
    for mode in modes:
        se = re.compile(mode).search(text)
        if se:
            value = se.groups()[0]
            return value
    return value


def field_judge(text, field, mode, filter_char):
    replace_str = "()-*&,./{}[]^%$#@！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    field_no_pun = ''.join([i for i in field if i not in list(replace_str)])
    field_int = [i for i in field_no_pun if i.isdigit()]
    if len(field_int) > 2:
        field = extract_mode(text, [mode[-1]], filter_char)
        field = field[1:].split(' ') if field.startswith(' ') else field.split(' ')[0]
    split_str = [':', '：', ',', '，']
    for i in split_str:
        if i in field[:int(len(field)/2)]:
            field = field.split(i)[-1]
    return field


class FT001001001001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT001001001001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', '*', ':']

        # with open(file_path, 'r', encoding='utf-8') as f:
        #     text = f.read()

        flg = [not re.search(ft_, text) for ft_ in self.mode['ft_mode']]
        if all(flg):
            return 'FT999'

        code = extract_mode(text, self.mode['code_mode'], filter_char).replace('：', '')  # 统一社会信用代码
        # self.no = self.extract_no()  # 注册号
        company = extract_mode(text, self.mode['company_mode'], filter_char+['：', '，', '；', ':'])  # 企业名称
        reg_money = extract_mode(text, self.mode['reg_money_mode'], filter_char).replace('：', '').replace('-', '')  # 注册资本
        rea_money = extract_mode(text, self.mode['rea_money_mode'], filter_char).replace('：', '')  # 实收资本
        address = extract_mode(text, self.mode['address_mode'], filter_char).replace('：', '')  # 住所
        lawyer = extract_mode(text, self.mode['lawyer_mode'], filter_char).replace('：', '').replace('、', '')  # 法定代表人
        type = extract_mode(text, self.mode['type_mode'], filter_char).replace('：', '').replace('，', '')  # 公司类型
        scope = self.extract_scope(text, self.mode['scope_mode'], filter_char+['：', '；'])  # 经营范围
        life = extract_mode(text, self.mode['life_mode'], filter_char).replace('：', '').replace('o', '0')  # 营业期限
        est_date = extract_mode(text, self.mode['est_date_mode'], filter_char).replace('：', '').replace('o', '0').replace('E', '日')  # 成立日期
        office = extract_mode(text, self.mode['office_mode'], filter_char).replace('：', '')  # 发证机关
        date = extract_mode(text, self.mode['date_mode'], filter_char).replace('：', '').replace('o', '0')  # 发证日期

        if company.endswith('公') or company.endswith('分公'):
            company += '司'
        if company.endswith('有l公司'):
            company = company.replace('有l公司', '有限公司')
        if scope.startswith('：'):
            scope = scope[1:]
        if re.search('营业', code):
            code = code[:code.index('营业')]
        if re.search('执照', code):
            code = code[:code.index('执照')]
        if re.search('注册号', company):
            company = code[:company.index('注册号')]
        if re.search('实', reg_money):
            reg_money = reg_money[:reg_money.index('实')]

        if reg_money.startswith('民币'):
            reg_money = '人' + reg_money

        if re.search('证照编号', scope):
            scope = re.sub('证照编号[0-9]+', '', scope)
        if re.search('企业标识', scope):
            scope = re.sub('企业标识[0-9]+', '', scope)
        if len(company) > 30 and '公司' in company:
            company = company[:company.index('公司')+2]

        code, company, reg_money, rea_money, address, lawyer, type, scope, life, est_date, office, date = \
            [i[1:].split(' ')[0] if i.startswith(' ') else i.split(' ')[0] for i in
             [code, company, reg_money, rea_money, address, lawyer, type, scope, life, est_date, office, date]]

        if company.isdigit():
            company = extract_mode(text, [self.mode['company_mode'][-1]], filter_char+['：', '，', '；', ':'])

        address_replace = "()-*&,./{}[]^%$#@！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        address_no_pun = ''.join([i for i in address if i not in list(address_replace)])
        if address_no_pun.isdigit():
            address = extract_mode(text, [self.mode['address_mode'][-1]], filter_char).replace('：', '')
            address = address[1:].split(' ')[0] if address.startswith(' ') else address.split(' ')[0]
        if len(address) > 30:
            if '室' in address:
                address = address[:address.index('室') + 1]
            elif len(address) > 40:
                if '.' in address:
                    address = address[:address.index('.')-1]
                elif len(address) > 50:
                    if '门' in address:
                        address = address[:address.index('门')+1]
                    elif '号' in address:
                        address = address[:address.index('号')+1]

        return {'证书名称': '营业执照', '所属主体名称': company, '统一社会信用代码': code, '企业名称': company,
                '注册资本': reg_money, '实收资本': rea_money, '住所': address, '法定代表人': lawyer, '公司类型': type,
                '经营范围': scope, '营业期限': life, '成立日期': est_date, '发证机关': office, '发证日期': date}

    def extract_scope(self, text, scope_mode, filter_char):
        """
         提取注册有效期限
        :return: str（注册有效期限）
        """
        value = ''
        for mode in scope_mode[1:]:
            vv = re.compile(mode).search(text[int(len(text)*0.3):].replace('\n', '').replace(' ', ''))
            if vv:
                return vv.groups()[0]

        for line in text.split('\n'):
            v = re.compile(scope_mode[0]).search(''.join([i for i in line if i not in filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value


class FT001001002001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT001001002001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_']

        # with open(file_path, 'r', encoding='utf-8') as f:
        #     text = f.read()

        flg = [not re.search(ft_, text) for ft_ in self.mode['ft_mode']]
        if all(flg):
            return 'FT999'

        code = extract_mode_full(text,  self.mode['code_mode'], filter_char)  # 代码
        name = extract_mode(text, self.mode['name_mode'], filter_char).replace('-', '')  # 机构名称
        type = extract_mode(text, self.mode['type_mode'], filter_char)  # 机构类型
        address = extract_mode(text, self.mode['address_mode'], filter_char)  # 地址
        expire = extract_mode(text, self.mode['expire_mode'], filter_char)  # 有效期
        no = extract_mode(text, self.mode['no_mode'], filter_char)  # 登记号
        office = extract_mode(text, self.mode['office_mode'], filter_char)  # 发证机关
        date = extract_mode(text, self.mode['date_mode'], filter_char)  # 发证日期
        lawyer = extract_mode(text, self.mode['lawyer_mode'], filter_char).replace('负责人', '')

        replace_str = [',', '。', ';', '；', ':', '，', '：', '、', '）', '（', '(', ')']
        for r_str in replace_str:
            code = code.replace(r_str, '')
            name = name.replace(r_str, '')
            type = type.replace(r_str, '')
            address = address.replace(r_str, '')
            expire = expire.replace(r_str, '')
            no = no.replace(r_str, '')
            office = office.replace(r_str, '')
            date = date.replace(r_str, '')
            lawyer = lawyer.replace(r_str, '')

        if expire.startswith(' '):
            expire = expire[1:]
        name = name.split(' ')[0]
        if len(name) > 20:
            if re.search('有限公', name):
                name = name[:name.index('有限公')+3]
        if name.endswith('限公') or name.endswith('分公'):
            name = name + '司'
        if lawyer.startswith(' '):
            lawyer = lawyer[1:]
        if re.compile('(企业法|业法人|企业结)').search(type):
            type = '企业法人'
        elif re.compile('(企业非|非法人)').search(type):
            type = '企业非法人'
        if re.search('企醒滋', type):
            type = '企业法人'
        name = name.replace('分公疆', '分公司')

        return {'证书名称': '组织机构代码证', '所属主体名称': name, '代码': code.split(' ')[0],
                '机构名称': name, '法定代表人': lawyer.split(' ')[0], '机构类型': type.split(' ')[0],
                '地址': address.split(' ')[0], '有效期': expire.split(' ')[0], '登记号': no.split(' ')[0],
                '发证机关': office.split(' ')[0], '发证日期': date}


class FT001001003001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT001001003001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_']

        # with open(file_path, 'r', encoding='utf-8') as f:
        #     text = f.read()

        flg = [not re.search(ft_, text) for ft_ in self.mode['ft_mode']]
        if all(flg):
            return 'FT999'

        no = extract_mode(text, self.mode['no_mode'], filter_char)
        name = extract_mode(text, self.mode['name_mode'], filter_char)
        legal = extract_mode(text, self.mode['legal_mode'], filter_char).replace(')', '').replace('）', '')
        address = extract_mode(text, self.mode['address_mode'], filter_char)
        type = extract_mode(text, self.mode['type_mode'], filter_char)
        scope = extract_mode_full(text, self.mode['scope_mode'], filter_char)
        app_office = extract_mode(text, self.mode['app_office_mode'], filter_char)
        withhold = extract_mode(text, self.mode['withhold_mode'], filter_char)
        cer_office = extract_mode(text, self.mode['cer_office_mode'], filter_char)
        date = extract_mode(text, self.mode['date_mode'], filter_char)

        replace_str = [',', '。', ';', '；', ':', '，', '：', '、']
        for r_str in replace_str:
            no = no.replace(r_str, '')
            name = name.replace(r_str, '')
            legal = legal.replace(r_str, '')
            address = address.replace(r_str, '')
            type = type.replace(r_str, '')
            # scope = scope.replace(r_str, '')
            app_office = app_office.replace(r_str, '')
            withhold = withhold.replace(r_str, '')
            cer_office = cer_office.replace(r_str, '')
            date = date.replace(r_str, '')

        name_int = [i for i in name if i.isdigit()]
        if len(name_int) > 2:
            name = extract_mode(text, [self.mode['name_mode'][-1]], filter_char)
            if ':' in name:
                name = name.split(':')[-1]
        if re.search('依法', withhold) or re.search('确定', withhold):
            withhold == '依法确定'
        if not withhold:
            withhold = '依法确定'

        if name.endswith('有限公'):
            name += '司'
        if legal.startswith(' '):
            legal = legal[1:]

        legal = legal.replace('地址', '')

        if app_office.startswith('商行政'):
            app_office = '工' + app_office
        if scope.count('地址') > 0 and scope.count('名称') > 0:
            scope = scope.replace('地址', '').replace('名称', '')

        return {'证书名称': '税务登记证', '所属主体名称': name.split(' ')[0], '证书号': no.split(' ')[0],
                '纳税人名称': name.split(' ')[0],
                '法定代表人': legal.split(' ')[0], '地址': address.split(' ')[0],
                '登记注册类型': type.split(' ')[0], '经营范围': scope, '批准机关': app_office.split(' ')[0],
                '扣缴义务': withhold.split(' ')[0], '发证机关': cer_office.split(' ')[0],
                '发证日期': date.split(' ')[0]}


class FT001002001001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT001002001001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', '。']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        flg = [not re.search(ft_, text) for ft_ in self.mode['ft_mode']]
        if all(flg):
            return 'FT999'

        code = extract_mode(text, self.mode['code_mode'], filter_char).replace(':', '').replace('：', '')  # 代码
        name = extract_mode(text, self.mode['name_mode'], filter_char).replace(':', '').replace('：', '')  # 企业名称
        address = extract_mode(text, self.mode['address_mode'], filter_char).replace(':', '').replace('：', '')  # 地址
        expire = extract_mode(text, self.mode['expire_mode'], filter_char).replace(':', '').replace('：', '')  # 有效期
        office = extract_mode(text, self.mode['office_mode'], filter_char).replace(':', '').replace('：', '')  # 发证机关
        date = extract_mode(text, self.mode['date_mode'], filter_char).replace(':', '').replace('：', '')  # 发证日期

        if re.search('中国', office) or re.search('征信', office) or re.search('中心', office):
            office = '中国人民银行征信中心'
        name = name.split(' ')[0]
        address = address.replace(' ', '')

        return {'证书名称': '机构信用代码证', '所属主体名称': name, '代码': code, '企业名称': name,
                '地址': address, '有效期': expire, '发证机关': office, '发证日期': date}


class FT001002002001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT001002002001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_']

        # with open(file_path, 'r', encoding='utf-8') as f:
        #     text = f.read()

        code = extract_mode(text, self.mode['code_mode'], filter_char)  # 代码
        name = extract_mode(text, self.mode['name_mode'], filter_char+[':', '：'])  # 单位名称
        address = extract_mode(text, self.mode['address_mode'], filter_char+[':', '：', '；', '）', ')'])  # 地址
        type = extract_mode(text, self.mode['type_mode'], filter_char).replace(':', '').replace('：', '').replace('，', '')  # 单位类型
        legal = extract_mode(text, self.mode['legal_mode'], filter_char).replace(':', '').replace('：', '').replace(')', '').replace('）', '')  # 法定代表人
        code1 = extract_mode(text, self.mode['code1_mode'], filter_char).replace(':', '').replace('：', '')  # 组织机构统一代码
        expire = extract_mode(text, self.mode['expire_mode'], filter_char).replace(':', '').replace('：', '')  # 有效期
        office = extract_mode(text, self.mode['office_mode'], filter_char).replace(':', '').replace('：', '')  # 发证机关
        date = extract_mode(text, self.mode['date_mode'], filter_char+[':', '：', '；', '）', ')'])  # 发证日期

        if name.startswith(' '):
            name = name[1:]
        if legal.startswith(' '):
            legal = legal[1:]
        if expire.startswith(' '):
            expire = expire[1:]
        if re.search('发证', expire):
            expire = expire[:expire.index('发证')]
        if code1.startswith(' '):
            code1 = code1[1:]
        name = field_judge(text, name, self.mode['name_mode'], filter_char)

        return {'证书名称': '社会保险登记证', '所属主体名称': name.split(' ')[0], '证书编号': code,
                '单位名称': name.split(' ')[0],
                '地址': address.split(' ')[0], '单位类型': type.split(' ')[0], '法定代表人': legal.split(' ')[0],
                '组织机构统一代码': code1.split(' ')[0],
                '有效期': expire.split(' ')[0], '发证机关': office, '发证日期': date}


class FT001002003001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT001002003001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', '|']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        company = extract_mode(text, self.mode['company_mode'], filter_char).replace(':', '').replace('：', '')  # 企业名称
        address = extract_mode(text, self.mode['address_mode'], filter_char).replace(':', '').replace('：', '')  # 企业地址
        type = extract_mode(text, self.mode['type_mode'], filter_char).replace(':', '').replace('：', '')  # 企业类型
        app_no = extract_mode(text, self.mode['app_no_mode'], filter_char).replace(':', '').replace('：', '')  # 批准号
        code = extract_mode(text, self.mode['code_mode'], filter_char).replace(':', '').replace('：', '')  # 进出口企业代码
        date = extract_mode(text, self.mode['company_mode'], filter_char).replace(':', '').replace('：', '')  # 发证日期
        app_date = extract_mode(text, self.mode['app_date_mode'], filter_char).replace(':', '').replace('：', '')  # 批准日期
        cer_no = extract_mode(text, self.mode['cer_no_mode'], filter_char).replace(':', '').replace('：', '')  # 发证序号
        life = extract_mode(text, self.mode['life_mode'], filter_char).replace(':', '').replace('：', '')  # 经营年限
        total = extract_mode(text, self.mode['total_mode'], filter_char).replace(':', '').replace('：', '')  # 投资总额
        reg = extract_mode(text, self.mode['reg_mode'], filter_char).replace(':', '').replace('：', '')  # 注册资本
        scope = extract_mode(text, self.mode['scope_mode'], filter_char).replace(':', '').replace('：', '')  # 经营范围
        title = extract_mode(text, self.mode['title_mode'], filter_char).replace(':', '').replace('：', '')  # 投资者标题
        office = extract_mode(text, self.mode['office_mode'], filter_char).replace(':', '').replace('：', '')  # 发证机关

        if company.endswith('公'):
            company += '司'
        if company.startswith('1') or company.startswith('I'):
            company = company[1:]
        if address.startswith('1') or address.startswith('I'):
            address = address[1:]
        if type.startswith('1') or type.startswith('I'):
            type = type[1:]
        if total.startswith('1') or total.startswith('I'):
            total = total[1:]
        if reg.startswith('1') or reg.startswith('I'):
            reg = reg[1:]
        if life.startswith('1') or life.startswith('I'):
            life = life[1:]
        if re.search('号', app_no):
            app_no = app_no[:app_no.index('号') + 1]
        if re.search('投', app_no):
            app_no = app_no[:app_no.index('投')]
        if re.search('外资企业', type):
            type = '外资企业'
        elif re.search('中外合资', type):
            type = '中外合资企业'

        type = type.replace('亚', '')
        life = life[1:].split(' ')[0] if life.startswith(' ') else life.split(' ')[0]

        return {'证书名称': '外商投资企业批准证书', '所属主体名称': company.split(' ')[0], '企业名称': company.split(' ')[0],
                '企业地址': address.split(' ')[0],
                '企业类型': type.split(' ')[0], '批准号': app_no, '进出口企业代码': code.split(' ')[0],
                '发证日期': date.split(' ')[0], '投资总额': total, '注册资本': reg.split(' ')[0],
                '经营范围': scope.split(' ')[0], '投资者标题': title, '发证机关': office, '批准日期': app_date.split(' ')[0],
                '发证序号': cer_no, '经营年限': life}


class FT001002004001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT001002004001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ',', '，', '：', ':']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        # flg = [not re.search(ft_, text) for ft_ in self.ft_mode]
        # if all(flg):
        #     return 'FT999'

        code = extract_mode(text, self.mode['code_mode'], filter_char)  # 核准号正则模式
        name = extract_mode(text, self.mode['name_mode'], filter_char)  # 企业名称正则模式
        legal = extract_mode(text, self.mode['legal_mode'], filter_char).replace(')', '').replace('）', '')  # 法定代表人正则模式
        bank = extract_mode(text, self.mode['bank_mode'], filter_char)  # 开户银行正则模式
        no = extract_mode(text, self.mode['no_mode'], filter_char)  # 账号正则模式
        office = extract_mode(text, self.mode['office_mode'], filter_char)  # 发证机关正则模式
        date = extract_mode(text, self.mode['date_mode'], filter_char)  # 发证日期正则模式
        number = extract_mode(text, self.mode['number_mode'], filter_char)

        if bank.startswith(' '):
            bank = bank[1:]
        if no.startswith(' '):
            no = no[1:]
        if name.startswith(' '):
            name = name[1:]
        if name.endswith('限公'):
            name += '司'
        if len(legal) > 10:
            legal = legal[:3]

        return {'证书名称': '银行开户许可证', '所属主体名称': name.split(' ')[0], '核准号': code.split(' ')[0],
                '证书编号': number, '企业名称': name.split(' ')[0],
                '法定代表人': legal.split(' ')[0], '开户银行': bank.split(' ')[0], '账号': no.split(' ')[0],
                '发证机关': office, '发证日期': date}


class FT001002005001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT001002005001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ',', '，', '：']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        # flg = [not re.search(ft_, text) for ft_ in self.ft_mode]
        # if all(flg):
        #     return 'FT999'

        name = extract_mode(text, self.mode['name_mode'], filter_char)  # 企业名称正则模式
        date = extract_mode(text, self.mode['date_mode'], filter_char)  # 发证日期正则模式

        if name.startswith(' '):
            name = name[1:]
        if name.endswith('限公'):
            name += '司'

        return {'证书名称': '外汇登记信息', '所属主体名称': name.split(' ')[0], '企业名称': name.split(' ')[0],
                '发证日期': date}


class FT001003110001:
    def __init__(self):
        self.filter_char = ['】', '【', '^', '”', '“', '_']
        self.not_con = [':', '：', '?', '？', '。']
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT001003110001']
        self.content = self.mode['content_mode']

    def __match(self, pattern, num):
        pattern_list = []
        for i in range(len(pattern)-num+1):
            pattern_list.append(pattern[i:i+num])
        pattern = '|'.join(pattern_list)
        return pattern

    def __confirm_ft(self, text):
        ft_001_mode = ['11111']
        ft_002_mode = [f'({self.__match("报关单位登记", 5)})', f'({self.__match("注册信息年度报告", 4)})']
        ft_003_mode = [f'({self.__match("高新技术企业", 5)})', '(企业名称.*?证书编号.*?\n.*?发证时间.*?有效期.*?\n.*?批准机关)']
        ft_005_mode = [f'({self.__match("对外贸易经营者", 4)})', f'({self.__match("经营者中英文类型", 4)})', '(个人财产)']
        ft_006_mode = [f'({self.__match("软件企业认定", 4)})', '(经评估)', f'({self.__match("进一步鼓励软件产业和集成电路", 5)})']
        ft_042_mode = [f'({self.__match("食品经营许可证", 5)})', '(食品经营者)', f'({self.__match("主体业态", 3)})', f'({self.__match("日常监督管理机构人员", 5)})']
        ft_084_mode = [f'({self.__match("安全生产许可证", 4)})', '(经济类型)', '(安全.*?证)', f'{self.__match("主要负责人", 4)}']
        ft_108_mode = [f'({self.__match("出入境检验检疫报检", 4)})', '(营业执照号)']
        ft_109_mode = [f'({self.__match("进口货物收发货人", 4)})']
        ft = 110
        ft_modes = {'001': ft_001_mode, '002': ft_002_mode, '003': ft_003_mode, '005': ft_005_mode, '006': ft_006_mode, '042': ft_042_mode, '084': ft_084_mode, '108': ft_108_mode, '109': ft_109_mode}
        for index, ft_model in ft_modes.items():
            for f_m in ft_model:
                if re.search(f_m, text):
                    print(f_m, re.search(f_m, text))
                    ft = index
                    return ft
        return ft

    def extract_info(self, file_path, page, FT, text):
        ft = self.__confirm_ft(text)
        if ft == '001':
            return self.__extract_ft_001(text)
        elif ft == '002':
            return self.__extract_ft_002(text)
        elif ft == '003':
            return self.__extract_ft_003(text)
        elif ft == '005':
            return self.__extract_ft_005(text)
        elif ft == '006':
            return self.__extract_ft_006(text)
        elif ft == '042':
            return self.__extract_ft_042(text)
        elif ft == '084':
            return self.__extract_ft_084(text)
        elif ft == '108':
            return self.__extract_ft_108(text)
        elif ft == '109':
            return self.__extract_ft_109(text)
        else:
            return self.__extract_ft_110(text)

    def __extract_ft_001(self, text):
        return {'证书名称': '外管局业务登记凭证', 'version': '001'}

    def __extract_ft_002(self, text):
        ft_002 = self.mode['002']
        num = extract_mode(text, ft_002['海关注册编码'], self.filter_char)
        code = extract_mode(text, ft_002['组织机构代码'], self.filter_char)
        name = extract_mode(text, ft_002['企业名称'], self.filter_char)
        address = extract_mode(text, ft_002['企业住所'], self.filter_char)
        apart_type = extract_mode(text, ft_002['企业经营类别'], self.filter_char)
        reg_date = extract_mode(text, ft_002['注册登记日期'], self.filter_char)
        lawyer = extract_mode(text, ft_002['法定代表人'], self.filter_char)
        expire = extract_mode(text, ft_002['有限期'], self.filter_char+[':', '：'])
        apart = extract_mode(text, ft_002['发证机关'], self.filter_char)
        date = extract_mode(text, ft_002['发证日期'], self.filter_char)
        num, code, name, address, apart_type, reg_date, lawyer, expire, apart, date = \
            [i[1:].split(' ')[0] if i.startswith(' ') else i.split(' ')[0] for i in
             [num, code, name, address, apart_type, reg_date, lawyer, expire, apart, date]]
        return {"证书名称": '海关报关单位注册登记证书', '海关注册编码': num, '组织机构代码': code, '企业名称': name,
                '企业住所': address, '企业经营类别': apart_type, '注册登记日期': reg_date, '法定代表人': lawyer,
                '有限期': expire, '发证机关': apart, '发证日期': date, 'version': '002'}

    def __extract_ft_003(self, text):
        ft_003 = self.mode['003']
        name = extract_mode(text, ft_003['企业名称'], self.filter_char)
        num = extract_mode(text, ft_003['证书编号'], self.filter_char)
        date = extract_mode(text, ft_003['发证日期'], self.filter_char)
        expire = extract_mode(text, ft_003['有效期'], self.filter_char+[':', '：'])
        apart = extract_mode(text, ft_003['发证机关'], self.filter_char)
        name, num, date, expire, apart = [i[1:].split(' ')[0] if i.startswith(' ') else i.split(' ')[0] for i in
             [name, num, date, expire, apart]]
        return {'证书名称': '高新技术企业证书', '企业名称': name, '证书编号': num, '发证日期': date,
                '有效期': expire, '发证机关': apart, 'version': '003'}

    def __extract_ft_005(self, text):
        ft_005 = self.mode['005']
        number = extract_mode(text, ft_005['备案登记表编号'], self.filter_char)
        in_code = extract_mode(text, ft_005['进出口企业代码'], self.filter_char)
        sc_code = extract_mode(text, ft_005['统一社会信用代码'], self.filter_char)
        man_zh = extract_mode(text, ft_005['经营者中文名称'], self.filter_char)
        man_en = extract_mode(text, ft_005['经营者英文名称'], self.filter_char)
        gr_code = extract_mode(text, ft_005['组织机构代码'], self.filter_char)
        man_type = extract_mode(text, ft_005['经营者类型'], self.filter_char)
        address = extract_mode(text, ft_005['住所'], self.filter_char)
        address_zh = extract_mode(text, ft_005['经营场所(中文)'], self.filter_char)
        address_en = extract_mode(text, ft_005['经营场所(英文)'], self.filter_char)
        apart = extract_mode(text, ft_005['备案机关'], self.filter_char)
        date = extract_mode(text, ft_005['发证日期'], self.filter_char)

        number, in_code, sc_code, man_zh, man_en, gr_code, man_type, address, address_zh, address_en, apart, date = [
            i[1:].split(' ')[0] if i.startswith(' ') else i.split(' ')[0] for i in [
                number, in_code, sc_code, man_zh, man_en, gr_code, man_type, address, address_zh, address_en, apart, date]]

        return {'证书名称': '对外贸易经营者备案登记表', '备案登记表编号': number, '进出口企业代码': in_code, '统一社会信用代码': sc_code,
                '经营者中文名称': man_zh, '经营者英文名称': man_en, '组织机构代码': gr_code, '经营者类型': man_type,
                '住所': address, '经营场所(中文)': address_zh, '经营场所(英文)': address_en, '备案机关': apart, '发证日期': date, 'version': '005'}

    def __extract_ft_006(self, text):
        ft_006 = self.mode['006']
        name = extract_mode(text, ft_006['企业名称'], self.filter_char)
        num = extract_mode(text, ft_006['证书编号'], self.filter_char)
        expire = extract_mode(text, ft_006['有效期'], self.filter_char)
        apart = extract_mode(text, ft_006['发证机关'], self.filter_char)
        date = extract_mode(text, ft_006['发证日期'], self.filter_char)

        name, num, expire, apart, date = [i[1:].split(' ')[0] if i.startswith(' ') else i.split(' ')[0] for i in [
            name, num, expire, apart, date]]
        return {'证书名称': '软件企认定证书', '企业名称': name, '证书编号': num, '有效期': expire, '发证机关': apart, '发证日期': date, 'version': '006'}

    def __extract_ft_042(self, text):
        ft_042 = self.mode['042']
        num = extract_mode(text, ft_042['许可证编号'], self.filter_char)
        name = extract_mode(text, ft_042['经营者名称'], self.filter_char)
        code = extract_mode(text, ft_042['社会信用代码'], self.filter_char)
        lawyer = extract_mode(text, ft_042['法定代表人'], self.filter_char)
        address = extract_mode(text, ft_042['住所'], self.filter_char)
        where = extract_mode(text, ft_042['经营场所'], self.filter_char)
        body = extract_mode(text, ft_042['主体业态'], self.filter_char)
        project = extract_mode(text, ft_042['经营项目'], self.filter_char)
        expire = extract_mode(text, ft_042['有效期'], self.filter_char)
        apart = extract_mode(text, ft_042['发证机关'], self.filter_char)
        man = extract_mode(text, ft_042['签发人'], self.filter_char)
        date = extract_mode(text, ft_042['发证日期'], self.filter_char)
        group= extract_mode(text, ft_042['日常监督管理机构'], self.filter_char)
        group_man = extract_mode(text, ft_042['日常监督管理人员'], self.filter_char)

        num, name, code, lawyer, address, where, body, project, expire, apart, man, group, group_man = [
            i[1:].split(' ')[0] if i.startswith(' ') else i.split(' ')[0] for i in [
                num, name, code, lawyer, address, where, body, project, expire, apart, man, group, group_man]]
        return {'证书名称': "食品经营许可证", '许可证编号': num, '经营者名称': name, '社会信用代码': code,
                '法定代表人': lawyer, '住所': address, '经营场所': where, '主体业态': body, '经营项目': project,
                '有效期': expire, '发证机关': apart, '签发人': man, '发证日期': date, '日常监督管理机构': group, '日常监督管理人员':group_man, 'version': '042'}

    def __extract_ft_084(self, text):
        ft_084 = self.mode['084']
        num = extract_mode(text, ft_084['证书编号'], self.filter_char)
        name = extract_mode(text, ft_084['企业名称'], self.filter_char)
        sec_type = extract_mode(text, ft_084['安全类型'], self.filter_char)
        expire = extract_mode(text, ft_084['有效期'], self.filter_char)
        apart = extract_mode(text, ft_084['发证机关'], self.filter_char)
        date = extract_mode(text, ft_084['发证日期'], self.filter_char)

        num, name, sec_type, expire, apart, date = [i[1:].split(' ')[0] if i.startswith(' ') else i.split(' ')[0] for i in [
            num, name, sec_type, expire, apart, date
        ]]

        return {'证书名称': '安全生产许可证', '证书编号': num, '企业名称': name, '安全类型': sec_type, '有效期': expire,
                '发证机关': apart, '发证日期': date, 'version': '084'}

    def __extract_ft_108(self, text):
        ft_108 = self.mode['108']

        num = extract_mode(text, ft_108['编号'], self.filter_char)
        record_type = extract_mode(text, ft_108['备案类别'], self.filter_char)
        record_num = extract_mode(text, ft_108['备案号码'], self.filter_char)
        name_zh = extract_mode(text, ft_108['企业名称中文'], self.filter_char)
        name_en = extract_mode(text, ft_108['企业名称英文'], self.filter_char)
        address = extract_mode(text, ft_108['住所'], self.filter_char)
        address_office = extract_mode(text, ft_108['营业场所'], self.filter_char)
        company_nature = extract_mode(text, ft_108['企业性质'], self.filter_char)
        company_type = extract_mode(text, ft_108['企业类别'], self.filter_char)
        business_license_num = extract_mode(text, ft_108['营业执照号'], self.filter_char)
        social_code = extract_mode(text, ft_108['统一社会信用代码'], self.filter_char)
        bank = extract_mode(text, ft_108['开户银行'], self.filter_char)
        lawyer = extract_mode(text, ft_108['法定代表人'], self.filter_char)
        apart = extract_mode(text, ft_108['备案机关'], self.filter_char)
        date = extract_mode(text, ft_108['发证日期'], self.filter_char)

        num, record_type, record_num, name_zh, name_en, address, address_office, company_nature, company_type, \
        business_license_num, social_code, bank, lawyer, apart, date = [i[1:].split(' ')[0] if i.startswith(' ')
                                                                                else i.split(' ')[0] for i in [
            num, record_type, record_num, name_zh, name_en, address, address_office, company_nature, company_type,
            business_license_num, social_code, bank, lawyer, apart, date]]

        return {'证书名称': '出入境检验检疫报检企业备案表', '编号': num, '备案类别': record_type, '备案号码': record_num,
                '企业名称中文': name_zh, '企业名称英文': name_en, '住所': address, '营业场所': address_office,
                '企业性质': company_nature, '企业类别': company_type, '营业执照号': business_license_num, '统一社会信用代码':
                social_code, '开户银行': bank, '法定代表人': lawyer, '备案机关': apart, '发证日期': date, 'version': '108'}

    def __extract_ft_109(self, text):
        ft_109 = self.mode['109']
        num = extract_mode(text, ft_109['海关注册登记编号'], self.filter_char)
        reg_date = extract_mode(text, ft_109['注册登记日期'], self.filter_char)
        name = extract_mode(text, ft_109['企业名称'], self.filter_char)
        address = extract_mode(text, ft_109['企业地址'], self.filter_char)
        address_business = extract_mode(text, ft_109['经营场所'], self.filter_char)
        lawyer = extract_mode(text, ft_109['法定代表人'], self.filter_char)
        money = extract_mode(text, ft_109['注册资本'], self.filter_char)
        scope = extract_mode_full(text, ft_109['经营范围'], self.filter_char).replace('经营范围', '')
        expire = extract_mode(text, ft_109['有效期'], self.filter_char)
        apart = extract_mode(text, ft_109['发证机关'], self.filter_char)
        date = extract_mode(text, ft_109['发证日期'], self.filter_char)

        num, reg_date, name, address, address_business, lawyer, money, scope, expire, apart, date = \
            [i[1:].split(' ')[0] if i.startswith(' ') else
             i.split(' ')[0] for i in [num, reg_date, name, address, address_business, lawyer, money, scope, expire, apart, date]]

        return {'证书名称': '海关进口货物收发货人报关注册登记证书', '海关注册登记编号': num, '注册登记日期': reg_date,
                '企业名称': name, '企业地址': address, '经营场所': address_business, '法定代表人': lawyer, '注册资本': money,
                '经营范围': scope, '有效期': expire, '发证机关': apart, '发证日期': date, 'version': '109'}

    def __extract_ft_110(self, text):
        mode = self.mode
        num = extract_mode(text, mode['no_mode'], self.filter_char)
        company = extract_mode(text, mode['company_mode'], self.filter_char)
        expire = extract_mode(text, mode['expire_mode'], self.filter_char)
        office = extract_mode(text, mode['office_mode'], self.filter_char)
        date = extract_mode(text, mode['date_mode'], self.filter_char)
        content = self.extract_content(text)

        num, company, expire, office, date, date, content = [i[1:].split(' ')[0] if i.startswith(' ')
                                                                                else i.split(' ')[0] for i in
                                                             [num, company, expire, office, date, date, content]]

        return {'证书名称': '资质证书', '证书编号': num, '企业名称': company, '发证机关': office, '发证日期': date, '许可内容': content, 'version': '110'}

    def extract_content(self, text):
        """
         提取注册有效期限
        :return: str（注册有效期限）
        """
        value = []
        key = []
        for line in text.split():
            for k, v in self.mode['content_mode'].items():
                rr = re.compile(v).search(''.join([i for i in line if i not in self.filter_char]))
                if rr:
                    value.append(''.join([t for t in rr.groups()[0] if t not in self.not_con]))
                    key.append(k)
        # value = dict(zip(key, value))
        if len(value) > 0:
            value = value[0]
        else:
            value = ''
        return value


class FT007001002001:
    def __init__(self):
        # global mode
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007001002001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        number = extract_mode(text, self.mode['number_mode'], filter_char).replace(':', '').replace('：', '').replace('；', '')
        class_ = extract_mode(text, self.mode['class_mode'], filter_char).replace('(', '').replace('（', '').replace(':', '').replace('：', '').replace('弟', '第')
        company = extract_mode(text, self.mode['company_mode'], filter_char).replace('、', '').replace('；', '').replace(':', '').replace('：', '')
        date_1 = extract_mode(text, self.mode['date_1_mode'], filter_char+['.', '，', ':', '：'])
        date_2 = extract_mode(text, self.mode['date_2_mode'], filter_char+['.', '，', ':', '：', ',', '。'])

        date_2 = date_2.replace('至', '')
        if company.endswith('公'):
            company += '司'
        # if date_1.endswith('H'):
        date_1 = date_1.replace('H', '日')
        # if date_2.endswith('H'):
        date_2 = date_2.replace('H', '日')
        date_2 = date_2.replace('54', '5').replace('64', '6')

        return {'证书名称': '商标注册证', '所属主体名称': company.split(' ')[0], '注册人': company.split(' ')[0],
                '注册号': number.split(' ')[0], '商标': '',
                '类别': class_.split(' ')[0], '注册日期': date_1.split(' ')[0],
                '有效期': date_2.split(' ')[0], '取得方式': '原始取得'}


class FT007002002001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007002002001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        name = extract_mode(text, self.mode['name_mode'], filter_char).replace(':', '').replace('：', '')  # 专利名称正则模式
        number = extract_mode(text, self.mode['number_mode'], filter_char).replace(':', '').replace('：', '')  # 证书号正则模式
        number_2 = extract_mode(text, self.mode['number_2_mode'], filter_char+[' ']).replace(':', '').replace('：', '')  # 专利号正则模式
        date = extract_mode(text, self.mode['date_mode'], filter_char).replace(':', '').replace('：', '')  # 申请日期正则模式
        company = extract_mode_full(text, self.mode['company_mode'], filter_char+[':', '：'])  # 专利权人正则模式
        class_ = extract_mode(text, self.mode['class_mode'], filter_char+[':', '：'])  # 专利类别

        if company.endswith('限公'):
            company += '司'
        if class_:
            if re.search('发', class_) or re.search('明', class_):
                class_ = '发明专利'
            elif re.search('外观', class_) or re.search('设计', class_):
                class_ = '外观设计专利'
            elif re.search('实用', class_) or re.search('新型', class_):
                class_ = '实用新型专利'
            else:
                class_ = '专利'
        else:
            class_ = '专利'

        replace_str = [',', '。', ';', '；', ':', '，', '：', '、']
        for r_str in replace_str:
            name = name.replace(r_str, '')
            number = number.replace(r_str, '')
            date = date.replace(r_str, '')
            number_2 = number_2.replace(r_str, '')
            company = company.replace(r_str, '')

        return {'证书名称': '专利证书', '所属主体名称': company.split(' ')[0], '专利名称': name.split(' ')[0],
                '证书号': number.split(' ')[0],
                '专利号': number_2.split(' ')[0], '申请日期': date.split(' ')[0],
                '专利类型': class_.split(' ')[0], '专利权人': company.split(' ')[0], '取得方式': '原始取得'}


class FT007003002001:
    def __init__(self):
        # global mode
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007003002001']
        # self.ft_mode = self.mode['ft_mode']

        self.name_mode = self.mode['name_mode']  # 软件名称正则模式

        self.name = ''

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', '*']

        # with open(file_path, 'r', encoding='utf-8') as f:
        #     text = f.read()

        self.name = self.extract_name(text, filter_char)  # 软件名称正则模式
        number = extract_mode(text, self.mode['number_mode'], filter_char)  # 著作权编号正则模式
        number_1 = extract_mode(text, self.mode['number_1_mode'], filter_char).replace('著作权', '')  # 登记号正则模式
        owner = extract_mode(text, self.mode['owner_mode'], filter_char)  # 著作权人正则模式
        date = extract_mode(text, self.mode['date_mode'], filter_char+['.', 'E'])  # 首次发表日期正则模式
        date_1 = extract_mode(text[int(len(text)*0.7):], self.mode['date_1_mode'], filter_char)  # 登记日期正则模式
        way = extract_mode(text, self.mode['way_mode'], filter_char)  # 取得方式正则模式

        for i in [['教', '软'], ['嗽', '软'], ['薯', ''], ['编号', ''], ['款', '']]:
            number = number.replace(i[0], i[1])

        replace_str = [',', '。', ';', '；', ':', '，', '：', '、', '-', '—']
        for r_str in replace_str:
            self.name = self.name.replace(r_str, '')
            number = number.replace(r_str, '')
            number_1 = number_1.replace(r_str, '')
            owner = owner.replace(r_str, '')
            date = date.replace(r_str, '')
            date_1 = date_1.replace(r_str, '')
            way = way.replace(r_str, '')

        if self.name.startswith(' '):
            self.name = self.name[1:]
        if re.search('原', way) or re.search('始', way):
            way = '原始取得'
        elif re.search('继', way):
            way = '继受取得'
        elif re.search('转让', way):
            way = '转让取得'
        else:
            way = '原始取得'

        return {'证书名称': '软件著作权登记证书', '所属主体名称': owner.split(' ')[0], '软件名称': self.name.split(' ')[0],
                '著作权编号': number.split(' ')[0],
                '登记号': number_1.split(' ')[0], '著作权人': owner.split(' ')[0],
                '首次发表日期': date.split(' ')[0], '登记日期': date_1.split(' ')[0], '取得方式': way.split(' ')[0]}

    def extract_name(self, text, filter_char):
        """
         提取软件名称
        :return: str（软件名称）
        """
        value = ''
        text_full = ''.join([i for i in text if i not in filter_char+[' ', '\n']])
        se = re.compile(self.name_mode[0]).search(text_full)
        if se:
            value = se.groups()[0]
        if len(value) > 40 or value == '':
            for i in self.name_mode[1:]:
                for line in text.split('\n'):
                    v = re.compile(i).search(''.join([i for i in line if i not in filter_char]))
                    if v:
                        value = v.groups()[0]
                        return value
        return value


class FT007003003001:
    def __init__(self):
        # global mode
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007003003001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', '*']

        reg_no = extract_mode(text, self.mode['reg_no_mode'], filter_char)
        name = extract_mode_full(text, self.mode['name_mode'], filter_char)
        type = extract_mode_full(text, self.mode['type_mode'], filter_char)
        producer = extract_mode_full(text, self.mode['producer_mode'], filter_char)
        owner = extract_mode_full(text, self.mode['owner_mode'], filter_char)
        end_time = extract_mode(text, self.mode['end_time_mode'], filter_char)
        release_time = extract_mode(text, self.mode['release_time_mode'], filter_char)
        reg_date = extract_mode(text, self.mode['reg_date_mode'], filter_char)
        date = extract_mode(text, self.mode['date_mode'], filter_char)
        office = extract_mode(text, self.mode['office_mode'], filter_char)

        office = '中华人民共和国国家版权局'

        return {'证书名称': "作品登记证书", '登记号': reg_no, '作品名称': name, '作品类别': type, '制作者': producer,
                '著作权人': owner, '创作完成时间': end_time, '首次公映时间': release_time, '登记日期': reg_date,
                '发证日期': date, '发证机关': office}


class FT007004002001:
    def __init__(self):
        # global mode
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007004002001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        name = extract_mode(text, self.mode['name_mode'], filter_char)  # 软件产品名称正则模式
        company = extract_mode(text, self.mode['company_mode'], filter_char)  # 申请企业正则模式
        class_ = extract_mode(text, self.mode['class_mode'], filter_char)  # 软件类别正则模式
        number = extract_mode(text, self.mode['number_mode'], filter_char)  # 证书编号正则模式
        date = extract_mode(text, self.mode['date_mode'], filter_char)  # 有效期正则模式
        date_1 = extract_mode(text, self.mode['date_1_mode'], filter_char).replace('-', '一').replace('—', '一')  # 发证日期正则模式

        replace_str = [',', '。', ';', '；', ':', '，', '：', '、']
        for r_str in replace_str:
            name = name.replace(r_str, '')
            company = company.replace(r_str, '')
            class_ = class_.replace(r_str, '')
            number = number.replace(r_str, '')
            date = date.replace(r_str, '')
            date_1 = date_1.replace(r_str, '')

        if class_:
            if class_.split('名')[0] == '发明':
                class_ = '发明专利'
            if class_.split('名')[0] == '实用新型':
                class_ = '实用新型发明专利'

        return {'证书名称': '软件产品登记证书', '所属主体名称': company.split(' ')[0], '软件产品名称': name.split(' ')[0],
                '申请企业': company.split(' ')[0],
                '软件类别': class_.split(' ')[0], '证书编号': number.split(' ')[0],
                '有效期': date.split(' ')[0], '发证日期': date_1.split(' ')[0], '取得方式': '原始取得'}


class FT007005001001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007005001001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_']

        # text = full_ocr(file_path, page, FT)
        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        number = extract_mode(text, self.mode['number_mode'], filter_char)  # 编号正则模式
        name = extract_mode(text, self.mode['name_mode'], filter_char)  # 域名正则模式
        owner = extract_mode(text, self.mode['owner_mode'], filter_char)  # 注册人正则模式
        date = extract_mode(text, self.mode['date_mode'], filter_char)  # 注册日期正则模式
        date_1 = extract_mode(text, self.mode['date_1_mode'], filter_char)  # 到期日期正则模式
        company = extract_mode(text, self.mode['company_mode'], filter_char)  # 发证单位正则模式
        date_2 = extract_mode(text, self.mode['date_2_mode'], filter_char)  # 有效期
        if date_2:
            if '至' in date_2:
                date, date_1 = date_2.split('至')
            else:
                date, date_2 = date_2[:int(len(date_2) / 2)], date_2[int(len(date_2) / 2) + 1:]

        number, name, owner, date, date_1, company, date_2 = [i[1:].split(' ')[0] if i.startswith(' ') else i.split(' ')[0] for i in [number, name, owner, date, date_1, company, date_2]]
        replace_str = [',', '。', ';', '；', ':', '，', '：', '）', ' ', ')']
        for r_str in replace_str:
            number = number.replace(r_str, '')
            name = name.replace(r_str, '')
            owner = owner.replace(r_str, '')
            company = company.replace(r_str, '')
        replace_str = [',', '。', '，', '）', ' ', ')']
        for r_str in replace_str:
            date = date.replace(r_str, '')
            date_1 = date_1.replace(r_str, '')
            date_2 = date_2.replace(r_str, '')

        date, date_1, date_2 = [i[1:] if i.startswith(':') or i.startswith('：') else i for i in [date, date_1, date_2]]

        return {'证书名称': '域名证书文件', '所属主体名称': owner, '编号': number, '域名': name,
                '注册人': owner, '注册日期': date,
                '到期日期': date_1, '发证单位': company}


class FT007006003001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007006003001']
        self.ft_mode = self.mode['ft_mode']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', '(', ')', '（', '）', ':', '：']
        filter_2_char = ['】', '【', '^', '”', '“', '_']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        flg = [not re.search(ft_, text) for ft_ in self.ft_mode]
        if all(flg):
            return 'FT999'

        number = extract_mode(text, self.mode['number_mode'], filter_char)  # 证书编号
        owner = extract_mode(text, self.mode['owner_mode'], filter_char)  # 土地使用权人
        area = extract_mode(text, self.mode['area_mode'], filter_char)  # 坐落
        number_1 = extract_mode(text, self.mode['number_1_mode'], filter_char)  # 地号
        way = extract_mode(text, self.mode['way_mode'], filter_2_char)  # 地类（用途）
        price = extract_mode(text, self.mode['price_mode'], filter_char)  # 取得价格
        class_ = extract_mode(text, self.mode['class_mode'], filter_char)  # 使用权类型
        date = extract_mode(text, self.mode['date_mode'], filter_char)  # 终止日期
        area_1 = extract_mode(text, self.mode['area_1_mode'], filter_char)  # 使用权面积
        area_2 = extract_mode(text, self.mode['area_2_mode'], filter_char)  # 独用面积
        area_3 = extract_mode(text, self.mode['area_3_mode'], filter_char)  # 分摊面积
        area_4 = extract_mode(text, self.mode['area_4_mode'], filter_char)  # 发证单位
        date_1 = extract_mode(text, self.mode['date_1_mode'], filter_char)  # 发证日期

        # 根据内容判断是否属于
        judge = [number, owner, area, number_1, way, price, class_, date,
                 area_1, area_2, area_3, area_4, date_1]
        flg_1 = [i == '' or not i for i in judge]
        if all(flg_1):
            return 'FT999'

        if re.search('号', number):
            number = number[:number.index('号') + 1]
        if owner.startswith('l') or owner.startswith('！'):
            owner = owner[1:]
        if area.startswith('l') or area.startswith('！'):
            area = area[1:]

        if area_1 != '':
            area_1 += '㎡'
        if area_2 != '':
            area_2 += '㎡'
        if area_3 != '':
            area_3 += '㎡'

        return {'证书名称': '国有土地使用证', '所属主体名称': owner, '证书编号': number, '土地使用权人': owner,
                '坐落': area, '地号': number_1,
                '地类（用途）': way, '取得价格': price, '使用权类型': class_, '终止日期': date,
                '使用权面积': area_1, '独用面积': area_2, '分摊面积': area_3, '发证单位': area_4,
                '发证日期': date_1}


class FT007006004001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007006004001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', '！', '：', '）', '（', ':']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        number = extract_mode_full_1(text, self.mode['number_mode'], filter_char)  # 编号
        owner = extract_mode_full_1(text, self.mode['owner_mode'], filter_char)  # 权利人
        location = extract_mode_full_1(text, self.mode['location_mode'], filter_char)  # 坐落
        share = extract_mode_full_1(text, self.mode['share_mode'], filter_char)  # 共有情况
        property_num = extract_mode_full_1(text, self.mode['property_num_mode'], filter_char)  # 不动产单元号
        right_type = extract_mode_full_1(text, self.mode['right_type_mode'], filter_char)  # 权利类型
        right_nature = extract_mode_full_1(text, self.mode['right_nature_mode'], filter_char)  # 权利性质
        purpose = extract_mode_full_1(text, self.mode['purpose_mode'], filter_char)  # 用途
        area = extract_mode_full_1(text, self.mode['area_mode'], filter_char).replace('m', '㎡')  # 面积
        service_time = extract_mode_full_1(text, self.mode['service_time_mode'], filter_char)  # 使用权限
        if purpose.startswith(' '):
            purpose = purpose[1:]
        if share.startswith(' '):
            share = share[1:]
        if right_type.startswith(' '):
            right_type = right_type[1:]

        # 根据内容判断是否属于
        judge = [number, owner, location, share, property_num, right_type,
                 right_nature, purpose, area, service_time]
        flg_1 = [i == '' or not i for i in judge]
        if all(flg_1):
            return 'FT999'

        return {'证书名称': '不动产权证', '所属主体名称': owner, '编号': number, '权利人': owner,
                '共有情况': share.split(' ')[0], '坐落': location, '不动产单元号': property_num,
                '权利类型': right_type.split(' ')[0], '权利性质': right_nature, '用途': purpose.split(' ')[0],
                '面积': area, '使用权限': service_time}


class FT007006005001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007006005001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ';', '：', ':']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        company = extract_mode(text, self.mode['company_mode'], filter_char)
        name = extract_mode(text, self.mode['name_mode'], filter_char)
        desti = extract_mode(text, self.mode['desti_mode'], filter_char)
        way = extract_mode(text, self.mode['way_mode'], filter_char)
        area = extract_mode(text, self.mode['area_mode'], filter_char)
        way_1 = extract_mode(text, self.mode['way_1_mode'], filter_char)
        company_1 = extract_mode_full_1(text, self.mode['company_1_mode'], filter_char)
        date = extract_mode_full_1(text, self.mode['date_mode'], filter_char)
        number = extract_mode(text, self.mode['number_mode'], filter_char)

        # 根据内容判断是否属于
        judge = [company, name, desti, way, area, way_1, company_1, date,
                 number]
        flg_1 = [i == '' or not i for i in judge]
        if all(flg_1):
            return 'FT999'

        # if re.search(' ', number):
        #     number = number.split(' ')[-1]

        return {'证书名称': '建设用地规划许可证', '所属主体名称': company, '用地单位': company,
                '用地项目名称': name.replace(' ', ''), '用地位置': desti, '用地性质': way, '用地面积': area,
                '建设规模': way_1, '发证单位': company_1, '发证日期': date, '编号': number}


class FT007006006001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007006006001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ';', '：', '）', '（', ':']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        number = extract_mode(text, self.mode['number_mode'], filter_char)
        company = extract_mode_full_1(text, self.mode['company_mode'], filter_char).replace(')', '').replace('）', '')
        name = extract_mode(text, self.mode['name_mode'], filter_char)
        way = extract_mode_full_1(text, self.mode['way_mode'], filter_char)
        company_1 = extract_mode_full_1(text, self.mode['company_1_mode'], filter_char)
        area = extract_mode_full_1(text, self.mode['area_mode'], filter_char)
        date = extract_mode_full_1(text, self.mode['date_mode'], filter_char)

        if company.startswith('l'):
            company = company[1:]
        if name.startswith('l'):
            name = name[1:]
        if area.startswith('l'):
            area = area[1:]

        # 根据内容判断是否属于
        judge = [number, company, name, way, company_1, area, date]
        flg_1 = [i == '' or not i for i in judge]
        if all(flg_1):
            return 'FT999'
        if len(company) > 30:
            company = ''

        return {'证书名称': '建设工程规划许可证', '所属主体名称': company, '编号': number, '建设单位': company,
                '建设项目名称': name, '建设位置': area, '建设规模': way, '发证机构': company_1, '发证日期': date}


class FT007006007001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007006007001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ':', '：']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        number = extract_mode(text, self.mode['number_mode'], filter_char)  # 编号
        company = extract_mode(text[:int(len(text)/2)], self.mode['company_mode'], filter_char).replace('）', '')  # 建设单位
        name = extract_mode_full_1(text, self.mode['name_mode'], filter_char)  # 建设项目名
        location = extract_mode_full_1(text, self.mode['location_mode'], filter_char)  # 建设位置
        scale = extract_mode_full_1(text, self.mode['scale_mode'], filter_char)  # 建设规模
        cer_company = extract_mode_full_1(text, self.mode['cer_company_mode'], filter_char)  # 发证机构
        cer_date = extract_mode_full_1(text, self.mode['cer_date_mode'], filter_char)  # 发证日期

        if company.startswith('I') or company.startswith('l'):
            company = company[1:]

        # 根据内容判断是否属于
        judge = [number, company, name, location, scale, cer_company, cer_date]
        if all([i == '' or not i for i in judge]):
            return 'FT999'

        return {'证书名称': '建设工程规划核实合格证', '所属主体名称': company, '编号': number,
                '建设单位': company, '建设项目名': name, '建设位置': location, '建设规模': scale,
                '发证机构': cer_company, '发证日期': cer_date}


class FT007006008001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007006008001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ':', '：']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        company = extract_mode(text, self.mode['company_mode'], filter_char)  # 建设单位
        name = extract_mode(text, self.mode['name_mode'], filter_char)  # 工程名称
        location = extract_mode_full_1(text, self.mode['location_mode'], filter_char)  # 建设地址
        scale = extract_mode(text, self.mode['scale_mode'], filter_char)  # 建设规模
        price = extract_mode_full_1(text, self.mode['price_mode'], filter_char)  # 合同价格
        design = extract_mode_full_1(text, self.mode['design_mode'], filter_char)  # 设计单位
        do_ = extract_mode_full_1(text, self.mode['do_mode'], filter_char)  # 施工单位
        superviser = extract_mode_full_1(text, self.mode['superviser_mode'], filter_char)  # 监理单位
        start = extract_mode(text, self.mode['start_mode'], filter_char)  # 合同开工日
        end = extract_mode(text, self.mode['end_mode'], filter_char)  # 合同竣工日
        cer_company = extract_mode(text, self.mode['cer_company_mode'], filter_char)  # 发证机关
        cer_date = extract_mode(text[int(len(text)*0.7):], self.mode['cer_date_mode'], filter_char)  # 发证日期

        # 根据内容判断是否属于
        judge = [company, name, location, scale, price, design, do_, superviser,
                 start, end, cer_company, cer_date]
        if all([i == '' or not i for i in judge]):
            return 'FT999'

        return {'证书名称': '建设工程施工许可证', '所属主体名称': company, '建设单位': company,
                '工程名称': name, '建设地址': location, '建设规模': scale, '合同价格': price,
                '设计单位': design, '施工单位': do_, '监理单位': superviser,
                '合同开工日': start, '合同竣工日': end, '发证机关': cer_company, '发证日期': cer_date}


class FT007006009001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007006009001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        sale = extract_mode_full_1(text, self.mode['sale_mode'], filter_char).replace(':', '').replace('：', '')  # 售房单位
        purpose = extract_mode_full_1(text, self.mode['purpose_mode'], filter_char).replace(':', '').replace('：', '')  # 用途
        location = extract_mode_full_1(text, self.mode['location_mode'], filter_char).replace(':', '').replace('：', '')  # 房屋坐落
        name = extract_mode_full_1(text, self.mode['name_mode'], filter_char).replace(':', '').replace('：', '')  # 项目名称
        scope = extract_mode_full_1(text, self.mode['scope_mode'], filter_char)  # 预售范围
        area = extract_mode_full_1(text, self.mode['area_mode'], filter_char).replace(':', '').replace('：', '')  # 建筑面积
        life = extract_mode_full_1(text, self.mode['life_mode'], filter_char).replace(':', '').replace('：', '')  # 土地使用年限
        cer_company = extract_mode_full_1(text, self.mode['cer_company_mode'], filter_char).replace(':', '').replace('：', '')  # 发证机关
        cer_date = extract_mode_full_1(text, self.mode['cer_date_mode'], filter_char).replace(':', '').replace('：', '')  # 发证日期

        # 根据内容判断是否属于
        judge = [sale, purpose, location, name, scope, area, life, cer_company,
                 cer_date]
        if all([i == '' or not i for i in judge]):
            return 'FT999'

        return {'证书名称': '商品房预售许可证', '所属主体名称': sale, '售房单位': sale, '用途': purpose,
                '房屋坐落': location, '项目名称': name.split(' ')[0], '预售范围': scope, '建筑面积': area,
                '土地使用年限': life, '发证机关': cer_company, '发证日期': cer_date}


class FT007006010001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007006010001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ';', ':', '：']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        area = extract_mode_full_1(text, self.mode['area_mode'], filter_char)
        number = extract_mode_full_1(text, self.mode['number_mode'], filter_char).replace('：', '').replace(':', '')
        name = extract_mode_full_1(text, self.mode['name_mode'], filter_char)
        company = extract_mode_full_1(text, self.mode['company_mode'], filter_char).replace('：', '')
        date = extract_mode_full_1(text, self.mode['date_mode'], filter_char)
        eas = extract_mode_full_1(text, self.mode['eas_mode'], filter_char)
        address = extract_mode_full_1(text, self.mode['address_mode'], filter_char)

        # 根据内容判断是否属于
        judge = [area, number, name, company, date, eas, address]
        if all([i == '' or not i for i in judge]):
            return 'FT999'
        if not area and eas:
            area = eas

        return {'证书名称': '建设工程竣工验收备案证', '所属主体名称': area, '企业名称': area,
                '备案编号': number, '项目名称': name, '发证机关': company, '发证日期': date,
                '建设单位': eas.split(' ')[0], '工程地点': address.split(' ')[0]}


class FT007006022001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007006022001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ';', '：', '！', ':', '|']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        flg = [not re.search(ft_, text.replace('\n', '')) for ft_ in self.mode['ft_mode']]
        if all(flg):
            return 'FT999'

        date = extract_mode_full_1(text, self.mode['date_mode'], filter_char)
        company = extract_mode_full_1(text, self.mode['company_mode'], filter_char)
        owner = extract_mode_full_1(text, self.mode['owner_mode'], filter_char)
        number = extract_mode_full_1(text, self.mode['number_mode'], filter_char)
        date_1 = extract_mode_full_1(text, self.mode['date_1_mode'], filter_char)
        location = extract_mode_full_1(text, self.mode['location_mode'], filter_char)
        way = extract_mode_full_1(text, self.mode['way_mode'], filter_char)
        way_1 = extract_mode_full_1(text, self.mode['way_1_mode'], filter_char)
        number_1 = extract_mode_full_1(text, self.mode['number_1_mode'], filter_char)
        area = extract_mode_full_1(text, self.mode['area_mode'], filter_char)
        date_2 = extract_mode_full_1(text, self.mode['date_2_mode'], filter_char)
        number_2 = extract_mode_full_1(text, self.mode['number_2_mode'], filter_char)
        area_1 = extract_mode_full_1(text, self.mode['area_1_mode'], filter_char)
        date_3 = extract_mode_full_1(text, self.mode['date_3_mode'], filter_char)

        if area.startswith(' '):
            area = area[1:]
        if way.startswith(' '):
            way = way[1:]
        if number_2.startswith(' '):
            number_2 = number_2[1:]

        # if re.search('博云路22号房产证', file_path):
        #     owner = '上海思华科技有限公司'
        #     location = '博云路22号1_4层'
        #     way_1 = '住宅、商业、工业'

        # 根据内容判断是否属于
        judge = [date, company, owner, number, date_1, location, way, way_1,
                 number_1, area, date_2, date_3, number_2]
        if all([i == '' or not i for i in judge]):
            return 'FT999'

        return {'证书名称': '房产权证', '所属主体名称': owner.split(' ')[0], '发证日期': date, '发证机关': company,
                '权利人': owner.split(' ')[0],
                '房地产权证编号': number, '登记日期': date_1, '房地坐落': location.split(' ')[0],
                '权属性质': way.split(' ')[0],
                '土地用途': way_1.split(' ')[0], '宗地号': number_1, '宗地面积': area.split(' ')[0],
                '使用期限': date_2, '幢号': number_2.split(' ')[0], '建筑面积': area_1, '竣工日期': date_3}


class FT007008001001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007008001001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', '[', ']', '|', ':', '：']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        owner = extract_mode_full_1(text, self.mode['owner_mode'], filter_char)  # 机动车所有人
        office = extract_mode_full_1(text, self.mode['office_mode'], filter_char)  # 登记机关
        date = extract_mode_full_1(text, self.mode['date_mode'], filter_char)  # 登记日期
        number = extract_mode_full_1(text, self.mode['number_mode'], filter_char)  # 登记编号

        for i in [':', '：', '。', '.', '】', '【', '^', '”', '“', '_', '[', ']', '|']:
            owner = owner.replace(i, '')
            office = office.replace(i, '')
            date = date.replace(i, '')
            number = number.replace(i, '')

        return {'证书名称': '车辆登记证', '所属主体名称': owner, '机动车所有人': owner,
                '登记机关': office, '登记日期': date, '机动车登记编号': number}


class FT007008002001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT007008002001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ':', '：']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()

        number = extract_mode_full_1(text, self.mode['number_mode'], filter_char)  # 车牌号
        owner = extract_mode_full_1(text, self.mode['owner_mode'], filter_char)  # 所有人
        type = extract_mode_full_1(text, self.mode['type_mode'], filter_char)  # 品牌型号
        re_num = extract_mode_full_1(text, self.mode['re_num_mode'], filter_char)  # 车辆识别代号
        reg_date = extract_mode(text, self.mode['reg_date_mode'], filter_char+['.'])  # 注册日期
        cer_date = extract_mode(text, self.mode['cer_date_mode'], filter_char+['.'])  # 发证日期

        if number.startswith(' '):
            number = number[1:]
        if owner.endswith('公'):
            owner += '司'
        if re_num.startswith(' '):
            re_num = re_num[1:]
        if type.startswith(' '):
            type = type[1:]

        return {'证书名称': '车辆行驶证', '所属主体名称': owner, '车牌号': number.split(' ')[0],
                '所有人': owner, '品牌型号': type.replace('Model', '').split(' ')[0],
                '车辆识别代号': re_num.split(' ')[0], '注册日期': reg_date, '发证日期': cer_date}


class FT011001001001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT011001001001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ':', '：', '，', "、", "。", '']

        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()
        # text = re.sub('[中华人民共和国]{4,}', '', text)
        text = re.sub('[中华人民共和国]{4,}', '', text)
        text = re.sub('[居民身份证]{4,}', '', text)
        text = re.sub('签发.*\n', '', text)
        text = re.sub('有效期.*\n', '', text)
        name = extract_mode_full_1(text, self.mode['name_mode'], filter_char)  # 姓名
        address = extract_mode_full(text, self.mode['address_mode'], filter_char)  # 住所
        no = extract_mode_full_1(text, self.mode['no_mode'], filter_char)  # 公民身份证
        sex = extract_mode_full_1(text, self.mode['sex_mode'], filter_char)  # 性别
        nation = extract_mode_full_1(text, self.mode['nation_mode'], filter_char)  # 民族
        date = extract_mode_full_1(text, self.mode['date_mode'], filter_char)  # 出生日期

        if '汉' in nation:
            nation = '汉'
        # try:
        #     if name == '' or address == '':
        #         print('name or address is None')
        #         pred = predict_ner(text.replace('\n', ''))
        #         print(pred)
        #         if name == '':
        #             name = pred['PER'][0] if pred['PER'] else ''
        #         if address == '':
        #             address = pred['LOC'][0] if pred['LOC'] else ''
        # except:
        #     pass

        name = ''.join([i for i in list(name) if i not in list('，,"“"”。')])
        if all([i == '' or not i for i in [name, address, no, sex, nation, date]]):
            return 'FT999'

        return {'证书名称': '身份证', '所属主体名称': name.split(' ')[0], '姓名': name.split(' ')[0], '住所': address,
                '公民身份证': no, '性别': sex, '民族': nation, '出生日期': date}


class FT012002001001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT012002001001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ':', '：', '，', "、", "。", '']

        no = extract_mode(text, self.mode['no_mode'], filter_char)
        name = extract_mode(text, self.mode['name_mode'], filter_char)
        expire = extract_mode(text, self.mode['expire_mode'], filter_char)
        permission = extract_mode(text, self.mode['permission_mode'], filter_char)
        cer_company = extract_mode(text, self.mode['cer_company_mode'], filter_char)
        date = extract_mode(text, self.mode['date_mode'], filter_char)

        if all([i == '' or not i for i in [name, expire, no, permission, cer_company, date]]):
            return 'FT999'

        return {'证书名称': '排污许可证', '所属主体名称': name.split(' ')[0], '单位名称': name.split(' ')[0], '证书编号': no,
                '许可内容': permission, '有效期': expire, '发证机关': cer_company, '发证日期': date}


class FT012002002001:
    def __init__(self):
        mode = pickle.load(open('infomodels/FT_REGULAR.json', 'rb'))
        mode = eval(str(base64.b64decode(mode), encoding='utf-8'))
        self.mode = mode['FT012002002001']

    def extract_info(self, file_path, page, FT, text):
        filter_char = ['】', '【', '^', '”', '“', '_', ':', '：', '，', "、", "。", '']

        no = extract_mode(text, self.mode['no_mode'], filter_char)
        name = extract_mode(text, self.mode['name_mode'], filter_char)
        expire = extract_mode(text, self.mode['expire_mode'], filter_char)
        permission = extract_mode(text, self.mode['permission_mode'], filter_char)
        cer_company = extract_mode(text, self.mode['cer_company_mode'], filter_char)
        date = extract_mode(text, self.mode['date_mode'], filter_char)

        if all([i == '' or not i for i in [name, expire, no, permission, cer_company, date]]):
            return 'FT999'

        return {'证书名称': '排水许可证', '所属主体名称': name.split(' ')[0], '企业名称': name.split(' ')[0], '证书编号': no,
                '许可内容': permission, '有效期': expire, '发证机关': cer_company, '发证日期': date}


# model = FT001002004001()
# result = model.extract_info(r'FT_test\FT001001001\1 国弘开元 营业执照（正本）.jpg', 1, 'FT001001001001')
# print(result)