import argparse
import os.path


class ArgumentManager:

    @staticmethod
    def make_dir(dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def get_train_args() -> argparse.Namespace:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()

        parser.add_argument('--data_dir', type=str, default='dataset\\pixel_art_tree\\train')
        parser.add_argument('--label_path', type=str, default='dataset\\labels.csv')
        parser.add_argument('--model_out_dir', type=str, default='results\\weights')
        ArgumentManager.make_dir('results\\weights')
        parser.add_argument('--fig_out_dir', type=str, default='results\\figures')
        ArgumentManager.make_dir('results\\figures')

        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--print_freq', type=int, default=1000)
        parser.add_argument('--disc_freq', type=int, default=1)
        parser.add_argument('--num_epoch', type=int, default=3150)
        parser.add_argument('--num_worker', type=int, default=8)
        parser.add_argument('--save_freq', type=int, default=1)
        parser.add_argument('--resume', type=str, default=None)

        parser.add_argument('--gen_lr', type=float, default=1e-4)
        parser.add_argument('--disc_lr', type=float, default=1e-4)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--beta2', type=float, default=0.9)
        parser.add_argument('--noise_len', type=int, default=128)
        parser.add_argument('--lambda_gp', type=int, default=10)

        args = parser.parse_args()

        print(args)

        return args

    @staticmethod
    def get_eval_args() -> argparse.Namespace:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()

        parser.add_argument('--noise_dim', type=int, default=128)
        parser.add_argument('--model_path', type=str, default='results\\best.pkl')
        parser.add_argument('--weight_key', type=str, default='model')
        parser.add_argument('--image_out_dir', type=str, default='results\\images')
        ArgumentManager.make_dir('results\\images')

        args = parser.parse_args()

        print(args)

        return args
