import argparse
import os


class Options:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--train_data_dir", type=str, default=None, help="Train directory folder path")
        self.parser.add_argument("--test_data_dir", type=str, default=None,  help="Test directory folder path")
        self.parser.add_argument("--image_size", type=int, default=256, help="Size to reshape the image, the image will be square, height equal to width")
        self.parser.add_argument("--augmentation", type=str, default="False", help="Do you want to augment training dataset? True/False")
        self.parser.add_argument("--augmentation_iterations", type=int, default=5, help="Number of augmentations iterations per image")
        self.parser.add_argument("--patches",  type=str, default="False", help="Do you want to exctract patches from images  True/False")
        self.parser.add_argument("--patch_size", type=int, default=32, help="Patch_size, the image will be square, height equal to width")
        self.parser.add_argument("--batch_size", type=int, default=32)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt

    def train_validate(self):

        opt = self.parse()
        valid = True

        # training folder validation
        try:
            if os.path.exists(opt.train_data_dir) and os.path.isdir(opt.train_data_dir):
                if not os.listdir(opt.train_data_dir):
                    print(f"Train diretory {opt.train_data_dir} is empty")
                    return False
            else:
                print(f"Train diretory {opt.train_data_dir}  don't exists")
                return False
        except:
            print(f"Train diretory {opt.train_data_dir}  don't exists")
            return False

        if opt.augmentation not in ["True", "False"]:
            print("Augmentation has to be True or False")
            return False

        if opt.augmentation == "True" and opt.augmentation_iterations < 2:
            print(f"Augmentation iterations had to be bigger than 1")
            return False

        if opt.patches not in ["True", "False"]:
            print("Patches has to be True or False")
            return False

        if opt.patches == "True" and opt.patch_size < 8:
            print("Patches size has to be >= 8")
            return False

        if(opt.image_size < 64):
            print("Image size has to be >= 64")
            return False

        if(opt.batch_size < 1):
            print("Batch size has to be >= 1")
            return False
       
        return valid
