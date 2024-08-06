import torch
from collections import OrderedDict
from future.utils import iteritems

def convert_weights(state_dict):
    tmp_weights = OrderedDict()
    for name, params in iteritems(state_dict):
        tmp_weights[name.replace('module.', '')] = params
    return tmp_weights

class Model(object):
    def __init__(self, args):
        self.args = args
        self.state = {}

    def set_train(self, model_list):
        for m in model_list:
            m.train()
    def set_eval(self, model_list):
        for m in model_list:
            m.eval()

    def log_write(self,log_str,log_path=None):
        print(log_str)
        if log_path is None:
            log_path=self.args.log_path
        if log_path is None :
            return
        if log_path != '':
            with open(log_path, 'a+') as writer:
                writer.write(log_str)

    def save_model(self, model, model_path):
        torch.save(model.state_dict(), model_path)

    def load_model(self, model, model_path):
        # model.load_state_dict(convert_weights(torch.load(model_path)))
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            model.load_state_dict(convert_weights(torch.load(model_path)))
        #model.module.load_state_dict(torch.load(model_path))
        return model

    def swa_init(self,model_list):
        if 'swa_list' not in self.state:
            log_str = 'SWA Initializing'
            self.log_write(log_str)
            swa_state_list = []
            for model in model_list:
                swa_state = {'models_num': 1}
                for n, p in model.named_parameters():
                    swa_state[n] = p.data.clone().detach()
                swa_state_list.append(swa_state)
            self.state['swa_list'] = swa_state_list

    def swa_step(self, model_list):
        if 'swa_list' in self.state:
            swa_state_list = self.state['swa_list']
            for i, swa_state in enumerate(swa_state_list):
                swa_state['models_num'] += 1
                beta = 1.0 / swa_state['models_num']
                model = model_list[i]
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        swa_state[n].mul_(1.0 - beta).add_(beta, p.data)

    def swap_swa_params(self, model_list):
        if 'swa_list' in self.state:
            swa_state_list = self.state['swa_list']
            for i, swa_state in enumerate(swa_state_list):
                model = model_list[i]
                for n, p in model.named_parameters():
                    p.data, swa_state[n] = swa_state[n], p.data

    def disable_swa(self):
        if 'swa_list' in self.state:
            del self.state['swa_list']

class ModelD(object):
    def __init__(self, args):
        self.args = args
        self.state = {}

    def set_train(self, model_list):
        for m in model_list:
            m.train()
    def set_eval(self, model_list):
        for m in model_list:
            m.eval()

    def log_write(self,log_str,log_path=None):
        if self.args.local_rank == 0:
            print(log_str)
            if log_path is None:
                log_path=self.args.log_path
            if log_path is None :
                return
            if log_path != '':
                with open(log_path, 'a+') as writer:
                    writer.write(log_str)

    def save_model(self, model, model_path):
        torch.save(model.state_dict(), model_path)

    def load_model(self, model, model_path):
        # model.load_state_dict(convert_weights(torch.load(model_path)))
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            model.load_state_dict(convert_weights(torch.load(model_path)))
        #model.module.load_state_dict(torch.load(model_path))
        return model

    def swa_init(self,model_list):
        if 'swa_list' not in self.state:
            log_str = 'SWA Initializing'
            self.log_write(log_str)
            swa_state_list = []
            for model in model_list:
                swa_state = {'models_num': 1}
                for n, p in model.named_parameters():
                    swa_state[n] = p.data.clone().detach()
                swa_state_list.append(swa_state)
            self.state['swa_list'] = swa_state_list

    def swa_step(self, model_list):
        if 'swa_list' in self.state:
            swa_state_list = self.state['swa_list']
            for i, swa_state in enumerate(swa_state_list):
                swa_state['models_num'] += 1
                beta = 1.0 / swa_state['models_num']
                model = model_list[i]
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        swa_state[n].mul_(1.0 - beta).add_(beta, p.data)

    def swap_swa_params(self, model_list):
        if 'swa_list' in self.state:
            swa_state_list = self.state['swa_list']
            for i, swa_state in enumerate(swa_state_list):
                model = model_list[i]
                for n, p in model.named_parameters():
                    p.data, swa_state[n] = swa_state[n], p.data

    def disable_swa(self):
        if 'swa_list' in self.state:
            del self.state['swa_list']
