import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# import collections
# import csv
# import os
# import time
import re
import copy
import tensorflow as tf
# from tqdm import tqdm
import numpy as np

# import fields_extraction.bert_ner.bert.modeling as modeling
# import fields_extraction.bert_ner.bert.optimization as optimization
# import fields_extraction.bert_ner.bert.tokenization as tokenization

import bert_ner.bert.modeling as modeling
import bert_ner.bert.optimization as optimization
import bert_ner.bert.tokenization as tokenization


def tokenize_data(text_list):
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, 
        do_lower_case=FLAGS.do_lower_case)

    data_list = []
    for i,_ in enumerate(text_list):
        text = text_list[i]

        if len(text) > FLAGS.max_seq_length-2:
            text = text[0:(FLAGS.max_seq_length-2)]

        tokens = ['[CLS]'] + [char for char in text] +['[SEP]']
        seq_length = len(tokens)

        input_ids = []
        for token in tokens:
            try:
                ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))[0]
            except:
                ids = 0
            input_ids.append(ids)

        input_mask = [1] * len(tokens)
        segment_ids = [0] * len(tokens)

        while len(input_ids) < FLAGS.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        data_list.append([input_ids, seq_length, input_mask, segment_ids, tokens])
        
    return data_list



def predict_ner(sentence):
    global ner_sess

    # sentence = 
    sentence_length = len(sentence)
    print('sentence_length', sentence_length)

    text_list = [sentence[i:i+FLAGS.max_seq_length-2] for i in range(0,sentence_length, FLAGS.max_seq_length-2)]
    pred_data_list = tokenize_data(text_list)
    pred_data_list_length = len(pred_data_list)

    y_pred_all = []
    for i in range(0, pred_data_list_length, FLAGS.predict_batch_size):
        batch_pred_data_idx = [idx for idx in range(i, i+FLAGS.predict_batch_size) if idx < pred_data_list_length]
        feed_dict = {}
        feed_dict["input_ids_pl:0"] = [pred_data_list[idx][0] for idx in batch_pred_data_idx]
        feed_dict["seq_length_pl:0"] = [pred_data_list[idx][1] for idx in batch_pred_data_idx]
        feed_dict["input_mask_pl:0"] = [pred_data_list[idx][2] for idx in batch_pred_data_idx]
        feed_dict["segment_ids_pl:0"] = [pred_data_list[idx][3] for idx in batch_pred_data_idx]
        batch_logits, trans = ner_sess.run(['loss/logits:0', 'transitions:0'], feed_dict=feed_dict)

        batch_y_pred = [tf.contrib.crf.viterbi_decode(logits, trans)[0] for logits in batch_logits]
        # y_pred_all += [i for pred in batch_y_pred for i in pred[1:-1]]
        y_pred_all = [int(pred) for idx in batch_pred_data_idx for pred in batch_y_pred[idx-i][1:pred_data_list[idx][1]-1]]


    y_pred_all_str = "".join([str(y) for y in y_pred_all])
    res = {}
    spans = [o.span() for o in re.finditer("45*", y_pred_all_str)]
    res['LOC'] = [sentence[span[0]:span[1]] for span in spans]
    res['LOC_idx'] = spans
    spans = [o.span() for o in re.finditer("01*", y_pred_all_str)]
    res['PER'] = [sentence[span[0]:span[1]] for span in spans]
    res['PER_idx'] = spans
    spans = [o.span() for o in re.finditer("23*", y_pred_all_str)]
    res['ORG'] = [sentence[span[0]:span[1]] for span in spans]
    res['ORG_idx'] = spans

    # remove entity that length less than some value
    for entity_tag, span_tag in [('ORG', 'ORG_idx'),('LOC', 'LOC_idx'),('PER', 'PER_idx')]:
        min_len = 6 if entity_tag == 'ORG' else 2
        entity_list, span_list = copy.copy(res[entity_tag]), copy.copy(res[span_tag])

        temp_entity_set = set()
        for entity, span in zip(entity_list, span_list):

            if len(entity) < min_len:
                res[entity_tag].remove(entity)
                res[span_tag].remove(span)

    # remove same values and its idx
    for entity_tag, span_tag in [('ORG', 'ORG_idx'),('LOC', 'LOC_idx'),('PER', 'PER_idx')]:
        entity_list, span_list = copy.copy(res[entity_tag]), copy.copy(res[span_tag])

        temp_entity_set = set()
        for entity, span in zip(entity_list, span_list):

            if entity in temp_entity_set:
                res[entity_tag].remove(entity)
                res[span_tag].remove(span)
            else:
                temp_entity_set.add(entity)

    # print('after', res)
    return res





class FLAGS:
    bert_config_file = 'bert_ner/chinese_L-12_H-768_A-12/bert_config.json'
    vocab_file = 'bert_ner/chinese_L-12_H-768_A-12/vocab.txt'
    ckpt_weights = 'bert_ner/ckpt/weights'
    ckpt_meta = 'bert_ner/ckpt/weights.meta'
    max_seq_length = 300
    do_lower_case = True
    is_training = False
    predict_batch_size = 128

    # 如果替换模型，这里主要要修改
    tag2id = {'B-PER': 0, 'I-PER': 1, 'B-ORG': 2, 'I-ORG': 3, 'B-LOC': 4, 'I-LOC': 5, 'O': 6, 'X': 7}
    
    num_labels = 8

FLAGS.id2tag = {v:k for k,v in FLAGS.tag2id.items()}

with tf.Graph().as_default():
    ner_sess = tf.Session()
    saver = tf.compat.v1.train.import_meta_graph(FLAGS.ckpt_meta)
    saver.restore(ner_sess, FLAGS.ckpt_weights)
    print('model restored.')

# test_out = predict_ner('思华科技（上海）股份有限公司董事会在今天上午召开，主持人是李斌。')
# print('test_out: ', test_out)
