import torch
from XAMP import model_XAMP, model_ESM2

xamp_t_full = torch.load('models/xamp_t.model.pth')
xamp_e_full = torch.load('models/xamp_e.model.pth')

torch.save(xamp_t_full.state_dict(), 'models/xamp_t.state_dict.pth')
torch.save(xamp_e_full.state_dict(), 'models/xamp_e.state_dict.pth')