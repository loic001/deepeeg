import torch

#
# class InputDictSelector(torch.nn.Module):
#     def __init__(self, module, select_key='x'):
#         super(InputDictSelector, self).__init__()
#         self.module = module
#         self.select_key = select_key
#
#         from models.eeg_net_braindecode import EEGNetBraindecode
#         model = EEGNetBraindecode(in_chans=chan, n_classes=2, input_time_length=input_time_length, final_conv_length='auto').create_network()
#         self.sub_module = model
#
#     def forward(self, *args, **kwargs):
#         return self.module.forward(kwargs[self.select_key])


from sklearn.manifold import TSNE


class ConditionalModule(torch.nn.Module):
    def __init__(self, condition, sub_module):
        super(ConditionalModule, self).__init__()
        self.condition = condition
        assert isinstance(sub_module, torch.nn.Module)
        self.sub_module = sub_module

    def forward(self, *args, **fit_params):
        condition =  self.condition(fit_params) if callable(self.condition) else self.condition
        if condition:
            print('runn')
            return self.sub_module.forward(*args, **fit_params)
        print(fit_params)
        return args[0]

    def __repr__(self):
        return self.__class__.__name__

class TSNEViewer(torch.nn.Module):
    #     """
    # Interactive plot with tsne algo
    #
    # Parameters
    # ----------
    # params: function
    #     Should accept variable number of objects of type
    #     `torch.autograd.Variable` to compute its output.
    # """
    def __init__(self, params=None):
        super(TSNEViewer, self).__init__()
        self.params = params
        self.c = 0

    def _forward(self, *args, **fit_params):
        print(self.c)
        print(fit_params)
        self.c = self.c +1
        # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        return args[0]
