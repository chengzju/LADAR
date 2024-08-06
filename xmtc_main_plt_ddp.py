import argparse
import random
from torch.utils.data import DataLoader
from utils import *
import math
from datasets.XMTCDataset import *
from models.Model import convert_weights
from models.MADModel_PLT import MADModel_PLT
from apex import amp
from networks.modules import *
import torch.distributed as dist
from tqdm import tqdm


def gen_true_label_list(data, label2id):
    true_label_list = []
    for one_data in tqdm(data):
        true_labels = list(set(one_data['labels']) & set(label2id.keys()))
        true_labels_id = [label2id[x] for x in true_labels]
        true_label_list.append(true_labels_id)
    return true_label_list

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='path for the data folders')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning_rate')
    parser.add_argument('--encoder_lr', default=1e-4, type=float, help='encoder_learning_rate')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--gpuid', default='0', type=str)
    parser.add_argument('--batch_size', default=0, type=int)
    parser.add_argument('--train_model', action='store_true')
    parser.add_argument('--test_model', action='store_true')
    parser.add_argument('--save_path')
    parser.add_argument('--bert_path', default='./bert/bert-base-uncased')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--maxlength', type=int, default=500)
    parser.add_argument('--encoder_weights', default=None)
    parser.add_argument('--seed', default=-1,type=int)
    parser.add_argument('--last_freeze_layer',default=10,type=int)
    parser.add_argument('--swa_warmup', default= -1, type=int)
    parser.add_argument('--try_num', default=0, type=int)
    parser.add_argument('--pos_weight', default=0, type=int)
    parser.add_argument('--sample_layer', type=int, default=5)
    parser.add_argument('--atten_num', type=int, default=30)
    parser.add_argument('--droprate', type=float, default=0.4)
    parser.add_argument('--feature_mode', type=str, default='cls')  # cls | avg
    parser.add_argument('--len_frag', type=int, default=100)
    parser.add_argument('--num_frag', type=int, default=5)
    parser.add_argument('--candidates_num', type=int, default=2000)
    parser.add_argument('--candidates_topk', type=int, default=50)
    parser.add_argument('--cluster_id', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--sample_test', action='store_true')
    parser.add_argument('--sample_test_size', type=int, default=10000)
    parser.add_argument('--ps_metric', action='store_true')
    parser.add_argument('--test_epoch', type=int, default=-1)
    parser.add_argument('--test_file_path', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    args = parser.parse_args()
    assert args.train_model != args.test_model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    dist.init_process_group(backend='nccl', rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)

    from transformers import BertTokenizer, BertModel, set_seed, AdamW, BertConfig

    dataset_name = args.data_dir.split('/')[-1]
    if args.seed < 0:
        random_seed = random.randint(1, 500)
        print(dataset_name, ' use random seed', random_seed)
    else:
        print(dataset_name, ' use specified seed', args.seed)
        random_seed = args.seed
    set_seed(random_seed)


    if args.train_model:
        model_type = 'xmtc_{}_{}_fz{}_swa{}_fp{}_id{}'.format(
            args.encoder_lr,
            args.lr,
            args.last_freeze_layer,
            args.swa_warmup,
            1 if args.fp16 else 0,
            args.try_num,
        )
    else:
        model_type = args.test_file_path

    args.model_type = model_type
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    args.tokenizer = tokenizer

    train_file_name = 'total_clf_train_data.json'
    test_file_name = 'total_clf_test_data.json'
    train_data_path = os.path.join(args.data_dir, train_file_name)
    test_data_path = os.path.join(args.data_dir, test_file_name)
    train_data = json.load(open(train_data_path, 'r'))
    val_data = json.load(open(test_data_path, 'r'))

    if args.sample_test and args.train_model:
        val_data = random.sample(val_data, args.sample_test_size)

    label_freq_path = os.path.join(args.data_dir, 'label_freq.json')
    label_freq = json.load(open(label_freq_path))
    label_freq_desc = sorted(label_freq.items(), key=lambda x: x[1], reverse=True)
    label_weight = [x[1] for x in label_freq_desc]
    label_index = [x[0] for x in label_freq_desc]
    labels = label_index
    label2id = {j: i for i, j in enumerate(labels)}

    train_labels = []
    if args.ps_metric:
        train_labels = [[label2id[x] for x in item['labels']] for item in train_data]

    # train_data = train_data[:100]
    # val_data = val_data[:100]

    group_y = load_group(dataset_name, args.cluster_id)
    train_dataset = XmtcDataset_PLT(
        args,
        data=train_data,
         tokenizer=tokenizer,
         label2id=label2id,
        maxlength=args.maxlength,
        mode='train',
        group_y=group_y,
        device = device
         )
    val_dataset = XmtcDataset_PLT(
        args,
        data=val_data,
         tokenizer=tokenizer,
         label2id=label2id,
        maxlength=args.maxlength,
        mode='val',
        group_y = group_y,
        device=device
        )
    group_y = train_dataset.get_group_y()
    label2group_path = os.path.join(args.data_dir, 'label2group_{}.json'.format(args.cluster_id))
    label2group = get_label2group(group_y, len(labels), label2group_path)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size = args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              shuffle=False,
                              drop_last=False,
                              sampler=train_sampler
                              )
    val_loader = DataLoader(val_dataset,
                             batch_size=args.batch_size ,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             shuffle=False,
                             drop_last=False)

    model_config = BertConfig.from_pretrained(args.bert_path)
    model_config.output_hidden_states = True
    encoder = BertModel.from_pretrained(args.bert_path, config=model_config).to(device)

    network_layer = MANetwork_PLT(args=args,
                                  hidden_size=encoder.config.hidden_size,
                                  label_num=len(labels),
                                  group_num=group_y.shape[0],
                                  atten_num=args.atten_num,
                                  group_y=group_y,
                                  label2group=label2group,
                                  device=device).to(device)

    network_granu =MANetwork_PLT(args=args,
                                 hidden_size=encoder.config.hidden_size,
                                 label_num=len(labels),
                                 group_num=group_y.shape[0],
                                 atten_num=args.atten_num,
                                 group_y=group_y,
                                 label2group=label2group,
                                 device=device).to(device)

    fp_list = [encoder, network_layer, network_granu]

    if args.train_model:
        if args.encoder_weights is not None:
            encoder.load_state_dict(convert_weights(torch.load(args.encoder_weights,map_location="cpu")))
        all_layers = ['layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6', 'layer.7', 'layer.8',
                      'layer.9', 'layer.10', 'layer.11', 'pooler']
        if args.last_freeze_layer < 0:
            unfreeze_layers = all_layers[:]
        else:
            unfreeze_layers = all_layers[args.last_freeze_layer+1:]

        for name, param in encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    model = MADModel_PLT(args)
    encoder, network_layer, network_granu = fp_list
    encoder = nn.parallel.DistributedDataParallel(encoder,
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank,
                                                  find_unused_parameters=True)
    network_layer = nn.parallel.DistributedDataParallel(network_layer,
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)
    network_granu = nn.parallel.DistributedDataParallel(network_granu,
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)

    if args.train_model:
        optimizer_grouped_parameters = [
            {'params': encoder.parameters(), 'lr': args.encoder_lr},
            {'params': network_layer.parameters(), 'lr': args.lr},
            {'params': network_granu.parameters(), 'lr': args.lr}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, )
        if args.fp16:
            fp_list, optimizer = amp.initialize(fp_list, optimizer, opt_level=args.fp16_opt_level)

        models = [network_layer, network_granu]

        model.train(encoder=encoder, model=models, optimizer=optimizer,
                    train_loader=train_loader, valid_loader=val_loader, label2id=label2id,
                    label_cnt=train_labels, device=device
                    )
    if args.test_model:
        save_path = os.path.join(args.save_path, model_type)
        postfix = '_{}'.format(args.test_epoch)
        encoder_weights = os.path.join(save_path,
                                       'BEST_encoder_checkpoint{}.pt'.format(postfix if args.test_epoch > 0 else ''))
        encoder = model.load_model(encoder, encoder_weights)
        layer_weights = os.path.join(save_path,
                                     'BEST_layer_checkpoint{}.pt'.format(postfix if args.test_epoch > 0 else ''))
        network_layer = model.load_model(network_layer, layer_weights)
        granu_weights = os.path.join(save_path,
                                     'BEST_granu_checkpoint{}.pt'.format(postfix if args.test_epoch > 0 else ''))
        network_granu = model.load_model(network_granu, granu_weights)
        models = [network_layer, network_granu]
        true_labels, outputs = model.predict(encoder, models, val_loader, device, 10)
        np.savez(os.path.join(save_path, 'outputs.npz'), y_true=true_labels, y_pred=outputs[1], y_prob=outputs[2])

        score = model.eval(true_labels, outputs[1], outputs[2], label2id, train_labels, final_eval=True)


if __name__ == '__main__':
    main()