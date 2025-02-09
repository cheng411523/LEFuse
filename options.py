import argparse

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # data loader related
        self.parser.add_argument('--dataroot', type=str, default='', help='path of data')
        
        
        # train
        # self.parser.add_argument('--batch_size', type=int, default='', help='batch size')
        # self.parser.add_argument('--lr', type=int, default=0.0001, help='Learning rate')
        # self.parser.add_argument('--total_epoch', type=int, default='')
        # self.parser.add_argument('--step_size', type=int, default='')
        # self.parser.add_argument('--gamma', type=int, default='')
        # self.parser.add_argument('--weight_decay', type=int, default=0)
        
            
        # test
        self.parser.add_argument('--ckpt_path', type=str, default="L2024.pth", help='Path to the pre-trained weights')
        self.parser.add_argument('--vi_path', type=str, default="./data/vi", help='Path to the Visible images')
        self.parser.add_argument('--ir_path', type=str, default="./data/ir", help='Path to the Infrared images')
        self.parser.add_argument('--out_path', type=str, default="./data/out", help='Path to save the Fusion results')
            
    
    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
