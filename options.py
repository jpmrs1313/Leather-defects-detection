import argparse
import os


class Options:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--train_data_dir", type=str, default=None, help="Train directory folder path")
        self.parser.add_argument("--test_data_dir", type=str, default=None,  help="Test directory folder path")
        self.parser.add_argument("--ground_truth_data_dir", type=str, default=None)
        self.parser.add_argument("--image_size", type=int, default=256, help="Size to reshape the image, the image will be square, height equal to width")
        self.parser.add_argument("--augmentation", type=str, default="False", help="Do you want to augment training dataset? True/False")
        self.parser.add_argument("--augmentation_iterations", type=int, default=5, help="Number of augmentations iterations per image")
        self.parser.add_argument("--batch_size", type=int, default=64)
        self.parser.add_argument("--patches",  type=str, default="False", help="Do you want to exctract patches from images  True/False")
        self.parser.add_argument("--patch_size", type=int, default=128, help="Patch_size, the image will be square, height equal to width")
        self.parser.add_argument("--loss", type=str, default="l2", help="Loss function l2/ssim")
    
    
    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt

  