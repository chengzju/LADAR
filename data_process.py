import argparse
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
import re
import json


def raw_tokenize(sentence:str, sep='/SEP/'):
    a=[token.lower() if token != sep else token for token in word_tokenize(sentence)
            if len(re.sub(r'[^\w]', '', token)) > 0]
    return a[:600]


def process_raw_text(data_dir, dataset_name):
    print(dataset_name, ' : raw process begin')
    input_file_list = ['train_raw_texts.txt', 'test_raw_texts.txt']
    output_file_list = ['train_texts.txt', 'test_texts.txt']
    data_path=data_dir
    for raw_text, output_text in zip(input_file_list, output_file_list):
        input_file=os.path.join(data_path, raw_text)
        output_file=os.path.join(data_path, output_text)
        if os.path.exists(output_file):
            print(dataset_name, output_file, 'exist, skip raw process')
            continue
        with open(input_file) as fp, open(output_file,'w') as fout:
            for line in tqdm(fp, desc='tokenizing raw text'):
                print(*raw_tokenize(line), file=fout)
    print(dataset_name, ' : raw process finish')


def load_data(data_input_file, label_input_file, data_output_file, label_output_file, l2s_output_file):
    data=[]
    freq4labels={}
    label2samples={}
    with open(data_input_file,'r') as f:
        for i, line in enumerate(tqdm(f)):
            one_text={'id':i, 'text':line.strip(), 'labels':[]}
            data.append(one_text)
    with open(label_input_file,'r') as f:
        for i, line in enumerate(tqdm(f)):
            one_label_list=line.strip().split()
            data[i]['labels']=one_label_list
            for l in one_label_list:
                freq4labels[l] = freq4labels.get(l, 0) +1
                l2s_list=label2samples.get(l,[])
                l2s_list.append(i)
                label2samples[l] = l2s_list
    if data_output_file is not None:
        json.dump(data, open(data_output_file,'w'),indent=2,ensure_ascii=False)
    if label_output_file is not None:
        json.dump(freq4labels, open(label_output_file,'w'),indent=2,ensure_ascii=False)
    if l2s_output_file is not None:
        json.dump(label2samples, open(l2s_output_file,'w'),indent=2,ensure_ascii=False)
    return data, freq4labels, label2samples


def clf_sentence_segment(data, tokenizer, maxlength=510, output=None):
    new_data = []
    max_token_length = 0
    for i, item in enumerate(tqdm(data)):
        d = {'id':item['id'], 'labels':item['labels']}
        text = item['text']
        input_ids = tokenizer.tokenize(text)
        input_ids = input_ids[:maxlength]
        d['input_ids'] = input_ids
        d['input_ids'] = tokenizer.convert_tokens_to_ids(d['input_ids'])
        max_token_length = max(max_token_length, len(d['input_ids']))
        new_data.append(d)
    print('max token length %d' % (max_token_length))
    if output is not None:
        json.dump(new_data, open(output, 'w'), indent=2, ensure_ascii=False)
    return new_data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='path for the data folders')
    parser.add_argument('--bert_path', default='./bert/bert-base-uncased' )
    parser.add_argument('--maxlength', type=int, default=500)
    parser.add_argument('--mask_rate', type=float, default=0.15)
    parser.add_argument('--gen_clf_total', action='store_true')
    args = parser.parse_args()

    dataset_name=args.data_dir.split('/')[-1]
    process_raw_text(args.data_dir, dataset_name)

    train_data_output_file=os.path.join(args.data_dir, 'train_data.json')
    label_output_file=os.path.join(args.data_dir, 'label_freq.json')
    train_l2s_output_file=os.path.join(args.data_dir, 'train_l2s.json')
    if os.path.exists(train_data_output_file) and os.path.exists(label_output_file) and os.path.exists(train_l2s_output_file):
        print(dataset_name,' train data exists.')
        train_data=json.load(open(train_data_output_file,'r'))
    else:
        print(dataset_name,' generate train data.')
        train_data_input_file=os.path.join(args.data_dir, 'train_texts.txt')
        train_label_input_file=os.path.join(args.data_dir, 'train_labels.txt')
        train_data, label_freq, train_l2s = load_data(train_data_input_file, train_label_input_file,
                                                      train_data_output_file, label_output_file,
                                                      train_l2s_output_file)
        print(dataset_name, 'finish generate train data.')

    test_data_output_file=os.path.join(args.data_dir,'test_data.json')
    test_l2s_output_file=os.path.join(args.data_dir,'test_l2s.json')
    if os.path.exists(test_data_output_file) and os.path.exists(test_l2s_output_file):
        print(dataset_name, ' test data exists.')
        test_data = json.load(open(test_data_output_file, 'r'))
        test_l2s = json.load(open(test_l2s_output_file, 'r'))
    else:
        print(dataset_name, ' generate test data.')
        test_data_input_file = os.path.join(args.data_dir, 'test_texts.txt')
        test_label_input_file = os.path.join(args.data_dir, 'test_labels.txt')
        test_data, _, test_l2s = load_data(test_data_input_file, test_label_input_file,
                                          test_data_output_file, None,
                                          test_l2s_output_file)
        print(dataset_name, 'finish generate test data.')

    if args.gen_clf_total:
        print(dataset_name,' generate CLF data for total')
        train_file_name = 'total_clf_train_data.json'
        test_file_name = 'total_clf_test_data.json'
        train_data4clf_base1_output_path = os.path.join(args.data_dir, train_file_name)
        test_data4clf_base1_output_path = os.path.join(args.data_dir, test_file_name)
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        train_data4clf = clf_sentence_segment(train_data,
                                              tokenizer,
                                              maxlength=args.maxlength,
                                              output=train_data4clf_base1_output_path,
                                              )
        test_data4clf = clf_sentence_segment(test_data,
                                             tokenizer,
                                             maxlength=args.maxlength,
                                             output=test_data4clf_base1_output_path,
                                             )
        print(dataset_name, ' finish generate CLF data')


if __name__ == '__main__':
    main()
