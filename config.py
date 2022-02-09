import argparse


class Options:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--train_data_dir", type=str, default=None)
        self.parser.add_argument("--test_data_dir", type=str, default=None)

        self.parser.add_argument("--augmentation", type=str, default="true",
                                 help="do you want to augment training dataset? true/false")
        self.parser.add_argument("--patches", type=str, default="true",
                                 help="do you want to exctract patches from images true/false?")
