from models.Model import Model
from tqdm import tqdm
from models.losses import *
from metric import *
from apex import amp

class MAModel(Model):
    def __init__(self,args):
        super(MAModel, self).__init__(args)


    def train(self, encoder, model, optimizer, train_loader, valid_loader, label2id, label_cnt, device):
        args = self.args
        save_path = os.path.join(args.save_path, args.model_type)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        log_path = os.path.join(save_path, 'train.log')
        args.log_path = log_path
        with open(log_path, 'w') as writer:
            writer.write(args.model_type)
        best_score = 0
        epochs_without_imp = 0
        best_epoch = 0

        network_layer, network_granu = model
        model_list = [encoder, network_layer, network_granu]
        for epoch in range(1, args.epochs+1):
            if epoch == args.swa_warmup:
                self.swa_init(model_list)
            log_str = '\nEpoch: %d/%d ' % (epoch, args.epochs)
            self.log_write(log_str)
            self.set_train(model_list)
            for batch_idx, inputs_list in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()

                labels, inputs_entire, inputs_frag = inputs_list
                labels = labels.to(device)
                input_ids_entire, input_type_ids_entire, input_mask_entire = [x.to(device) for x in inputs_entire]
                input_ids_list_frag, input_type_ids_list_frag, input_mask_list_frag = [x.to(device) for x in inputs_frag]
                bs, num_frag, len_frag = input_ids_list_frag.shape[:3]
                input_ids_list_frag = input_ids_list_frag.view(-1, len_frag)
                input_type_ids_list_frag = input_type_ids_list_frag.view(-1, len_frag)
                input_mask_list_frag = input_mask_list_frag.view(-1, len_frag)

                outs_list_entire = encoder(input_ids=input_ids_entire,
                                          attention_mask=input_mask_entire,
                                          token_type_ids=input_type_ids_entire)
                outs_entire = outs_list_entire[-1]
                outs_list_frag = encoder(input_ids=input_ids_list_frag,
                                          attention_mask=input_mask_list_frag,
                                          token_type_ids=input_type_ids_list_frag)
                outs_frag = outs_list_frag[-1]

                if args.feature_mode == 'cls':
                    features_layer = torch.stack([outs_entire[-i][:, 0] for i in range(1, args.sample_layer + 1)], dim=1)
                    feature_entire_last = outs_entire[-1][:, 0]

                    features_frag = outs_frag[-1][:, 0]
                    features_frag = features_frag.view(bs, num_frag, -1)
                elif args.feature_mode == 'avg':
                    features_layer = torch.stack([outs_entire[-i] for i in range(1, args.sample_layer + 1)], dim=1)
                    features_layer = torch.mean(features_layer, dim=2)
                    feature_entire_last = torch.cat([outs_entire[-i] for i in range(1, 2)], dim=-1)
                    feature_entire_last = torch.mean(feature_entire_last, dim=1)

                    features_frag = torch.cat([outs_frag[-i] for i in range(1, 2)], dim=-1)
                    features_frag = torch.mean(features_frag, dim=1)
                    features_frag = features_frag.view(bs, num_frag, -1)
                out_dict_layer = network_layer(features_layer)
                logits_layer = out_dict_layer['out']

                feature_entire_last = torch.unsqueeze(feature_entire_last, dim=1)
                features_granu = torch.cat((feature_entire_last, features_frag), dim=1)
                out_dict_granu = network_granu(features_granu)
                logits_granu = out_dict_granu['out']

                loss_layer = clf_loss(logits_layer, labels)
                loss_granu = clf_loss(logits_granu, labels)
                loss = loss_layer + loss_granu

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

                optimizer.param_groups[0]['lr'] = args.encoder_lr
                optimizer.param_groups[1]['lr'] = args.lr

            self.swa_step(model_list)
            self.swap_swa_params(model_list)
            outputs = self.predict(encoder, model, valid_loader, device, 10)
            score = self.eval(outputs[0], outputs[1],outputs[3], label2id, label_cnt)

            if score >= best_score:
                self.save_model(encoder, save_path + "/BEST_encoder_checkpoint.pt")
                self.save_model(network_layer, save_path + "/BEST_layer_checkpoint.pt")
                self.save_model(network_granu, save_path + "/BEST_granu_checkpoint.pt")
                np.savez(os.path.join(save_path, 'outputs.npz'), y_true=outputs[0], y_pred=outputs[1], y_prob=outputs[2])
                log_str = "\nNow the best lm score is %.6f, it was %.6f\n" % (score, best_score)
                self.log_write(log_str)
                best_score = score
                best_epoch = epoch
                epochs_without_imp = 0
            else:
                epochs_without_imp += 1
                log_str = "\nBest score is still %.6f," \
                          "best_epoch is %d," \
                          "epochs without imp. %d\n" % ( best_score, best_epoch,epochs_without_imp)
                self.log_write(log_str)
                if epoch - best_epoch > args.patience:
                    print("Early stopping")
                    return
            self.swap_swa_params(model_list)

    def predict(self, encoder, model, test_loader, device, k=5):
        args = self.args
        network_layer, network_granu = model
        model_list = [encoder, network_layer, network_granu]
        self.set_eval(model_list)
        outputs = [[], [], [], []]

        with torch.no_grad():
            for inputs_list in tqdm(test_loader):
                labels, inputs_entire, inputs_frag = inputs_list
                labels = labels.to(device)
                input_ids_entire, input_type_ids_entire, input_mask_entire = [x.to(device) for x in inputs_entire]
                input_ids_list_frag, input_type_ids_list_frag, input_mask_list_frag = [x.to(device) for x in
                                                                                       inputs_frag]
                bs, num_frag, len_frag = input_ids_list_frag.shape[:3]
                input_ids_list_frag = input_ids_list_frag.view(-1, len_frag)
                input_type_ids_list_frag = input_type_ids_list_frag.view(-1, len_frag)
                input_mask_list_frag = input_mask_list_frag.view(-1, len_frag)

                outs_list_entire = encoder(input_ids=input_ids_entire,
                                           attention_mask=input_mask_entire,
                                           token_type_ids=input_type_ids_entire)
                outs_entire = outs_list_entire[-1]
                outs_list_frag = encoder(input_ids=input_ids_list_frag,
                                         attention_mask=input_mask_list_frag,
                                         token_type_ids=input_type_ids_list_frag)
                outs_frag = outs_list_frag[-1]

                if args.feature_mode == 'cls':
                    features_layer = torch.stack([outs_entire[-i][:, 0] for i in range(1, args.sample_layer + 1)],
                                                 dim=1)
                    feature_entire_last = outs_entire[-1][:, 0]

                    features_frag = outs_frag[-1][:, 0]
                    features_frag = features_frag.view(bs, num_frag, -1)
                elif args.feature_mode == 'avg':
                    features_layer = torch.stack([outs_entire[-i] for i in range(1, args.sample_layer + 1)], dim=1)
                    features_layer = torch.mean(features_layer, dim=2)
                    feature_entire_last = torch.cat([outs_entire[-i] for i in range(1, 2)], dim=-1)
                    feature_entire_last = torch.mean(feature_entire_last, dim=1)

                    features_frag = torch.cat([outs_frag[-i] for i in range(1, 2)], dim=-1)
                    features_frag = torch.mean(features_frag, dim=1)
                    features_frag = features_frag.view(bs, num_frag, -1)
                out_dict_layer = network_layer(features_layer)
                logits_layer = out_dict_layer['out']

                feature_entire_last = torch.unsqueeze(feature_entire_last, dim=1)
                features_granu = torch.cat((feature_entire_last, features_frag), dim=1)
                out_dict_granu = network_granu(features_granu)
                logits_granu = out_dict_granu['out']

                logits = logits_layer + logits_granu

                if k == -1:
                    k = logits.shape[1]
                labels = labels.data.cpu().numpy()
                prob, pred = torch.topk(logits, k)
                prob = torch.sigmoid(prob).data.cpu().numpy()
                pred = pred.data.cpu().numpy()
                logits = logits.data.cpu().numpy()
                outputs[0].append(labels)
                outputs[1].append(pred)
                outputs[2].append(prob)
                outputs[3].append(logits)

        outputs = [np.concatenate(x, axis=0) for x in outputs]
        return outputs

    def eval(self, y_true, y_pred, y_logits, label2id, train_labels, final_eval=False):
        y_true_list = label2list(y_true)
        p1, p3, p5, n1, n3, n5 = base1_metric(y_true_list, y_pred, np.arange(len(label2id)))
        psp1, psp3, psp5, psn1, psn3, psn5 = ps_metric(y_true_list, y_logits, np.arange(len(label2id)), train_labels)

        log_str1 = '\n' + '\t'.join(['%.6f'] * 6)
        log_str1 = log_str1 % (p1, p3, p5, n1, n3, n5)
        log_str2 = '\n' + '\t'.join(['%.6f'] * 6)
        log_str2 = log_str2 % (psp1, psp3, psp5, psn1, psn3, psn5)
        if not final_eval:
            self.log_write(log_str1+log_str2)
        else:
            print(log_str1+log_str2)
        return n5




