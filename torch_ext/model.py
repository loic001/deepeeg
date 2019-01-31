# import torch
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
