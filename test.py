import os
import numpy as np
import torch
from skimage.io import imsave



from options import  TrainOptions 
from img_utils import RGB2YCrCb,YCbCr2RGB,image_read_cv2
from net import LEFuse

def test(ckpt,vi_path,ir_path,out_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fuser = LEFuse().to(device)
    fuser.load_state_dict(torch.load(ckpt)['model'])
    fuser.eval()

    FlagRGB = True # Whether to output RGB images
    
    with torch.no_grad():
        for img_name in os.listdir(vi_path):
            vi = image_read_cv2(os.path.join(vi_path, img_name), mode='RGB')[np.newaxis, ...] / 255.0
            vi = np.transpose(vi, (0, 3, 1, 2))  
            vi = torch.FloatTensor(vi)
            
            ir = image_read_cv2(os.path.join(ir_path, img_name), mode='RGB')[np.newaxis, ...] / 255.0
            ir = np.transpose(ir, (0, 3, 1, 2))  
            ir = torch.FloatTensor(ir)
            vi, ir = vi.cuda(), ir.cuda()
            
            vi_y,cr,cb = RGB2YCrCb(vi)
            ir_y,_,_ = RGB2YCrCb(ir)

            out = fuser(vi_y,ir_y)
            if FlagRGB:
                out = (out - torch.min(out)) / (torch.max(out) - torch.min(out))
                out = YCbCr2RGB(out,cb,cr)
                out = (out * 255.0).cpu().numpy().squeeze(0).astype('uint8')
                out = np.transpose(out, (1, 2, 0))
                imsave(os.path.join(out_path, "{}.png".format(img_name.split(sep='.')[0])),out)
            else:
                out = (out - torch.min(out)) / (torch.max(out) - torch.min(out))
                out = (out * 255.0).cpu().numpy().squeeze(0).squeeze(0).astype('uint8') 
                imsave(os.path.join(out_path, "{}.png".format(img_name.split(sep='.')[0])),out)


if __name__ == "__main__":
    parser = TrainOptions()
    opts = parser.parse()
    test(opts.ckpt_path,opts.vi_path,opts.ir_path,opts.out_path)