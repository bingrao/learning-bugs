from argparse import ArgumentParser
import torch


def get_main_argument(desc='Train Transformer'):
    parser = ArgumentParser(description=desc)
    # Command Project Related Parameters
    parser.add_argument('--project_name', type=str, required=True, default="")
    parser.add_argument('--project_raw_dir', type=str, required=True, default="")
    parser.add_argument('--project_processed_dir', type=str, required=True, default="")
    parser.add_argument('--project_config', type=str, required=True, default="")
    parser.add_argument('--project_log', type=str, required=True, default="")
    parser.add_argument('--project_save_config', type=bool, default=False)
    parser.add_argument('--project_checkpoint', type=str, default="")
    parser.add_argument('--phase', type=str, required=True, choices=['preprocess', 'train', 'predict', 'val'], default='train')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device_id', type=list, default=[0])
    # Predict input source sentence
    parser.add_argument('--source', type=str, default="I am a chinese.")
    parser.add_argument('--save_result', type=str, default=None)
    parser.add_argument('--debug', type=str, default="False")
    parser.add_argument('--position_style', type=str, choices=['default', 'sequence', 'tree', 'path'], default='default')

    args = parser.parse_args()

    config = vars(args)
    return config
