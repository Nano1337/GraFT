import argparse
import yaml
import os

def get_cfgs(): 
    """This function is used to parse the command line arguments and the config file
    
    :return cfgs: the parsed command line arguments and config file
    """
    config_parser = parser = argparse.ArgumentParser(description="Training Config", add_help=False)

    parser.add_argument('-c', '--config', type=str, default='cfgs/default.yaml', 
                        metavar='FILE', help='Path to config file')
    
    parser = argparse.ArgumentParser("MM-Mafia Training Script", add_help=False)

    # General parameters
    parser.add_argument('--output_dir', type=str, default='/home/ubuntu/mm_public/output_dir', help='Path to output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'val'], help='Phase of training')
    parser.add_argument('--cfg_name', type=str, default='default', help='Name of config file')
    parser.add_argument('--use_mixed_precision', type=bool, default=False, help='Whether to use mixed precision')
    parser.add_argument('--use_optuna', type=bool, default=False, help='Whether to use optuna')

    # Dataset parameters
    parser.add_argument('--dataroot', '--dataroot', type=str, default='/home/ubuntu/datasets', help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='bifrost/cls_4classes_v0_day_no_debris_marine', 
                        choices=[
                            'bifrost/cls_111classes_v0', 
                            'bifrost/cls_6classes_v0', 
                            'bifrost/cls_4classes_v0_day_no_debris_marine',
                            'bifrost/cls_4classes_v0_night_no_debris_marine',
                            ], help='Dataset to use')
    parser.add_argument('--dataset_name', type=str, default='BIFROST_CLS4_DAY', 
                        choices=[
                            'BIFROST_CLS111',
                            'BIFROST_CLS6',
                            'BIFROST_CLS4_DAY',
                            'BIFROST_CLS4_NIGHT',
                            ], help='Dataset to use')
    parser.add_argument('--data_augmentation', type=bool, default=False, help='Whether to use data augmentation')
    parser.add_argument('--augment_type', type=str, default='random_resize_crop', help='Type of data augmentation')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--input_size', nargs='+', type=int, default=[224, 224], help='Input size of images')
    parser.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip probability')
    parser.add_argument('--resize_min', type=float, default=0.2, help='Minimum resize ratio')
    parser.add_argument('--resize_max', type=float, default=1.0, help='Maximum resize ratio')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training vs validation')
    parser.add_argument('--one_hot', type=bool, default=True, help='Whether to use one hot encoding for labels')
    parser.add_argument('--image_num_for_reid_train', type=int, default=1, help='How many images (TRAIN) to sample from data')
    parser.add_argument('--image_num_for_reid_validate', type=int, default=25, help='How many images (VALIDATE) to sample from data')
    parser.add_argument('--num_triplet_samples', type=int, default=1, help='Number of triplet samples to use for training')

    parser.add_argument('--loss_scaling', type=bool, default=False, help='loss_scaling is bool')


    # Tpken Switch parameter
    parser.add_argument('--use_token', type=bool, default=False, help='Whether to use global cls tokens')
    parser.add_argument('--only_use_tokens_for_decoder', type=bool, default=False, help='Whether to use global cls tokens only for decoder')

    # MAE parameters
    parser.add_argument('--patch', type=bool, default=False, help='Whether to pre-patch the dataset for MAE')
    parser.add_argument('--model_patch_size', type=int, default=48, help='Size of individual patch for MAE')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Percentage of patches randomly masked in MAE')

    # Model parameters
    parser.add_argument('--model_modalities', nargs='+', type=str, default=['rgb', 'ir'], help='List of modalities')
    parser.add_argument('--model_rgb_pretrained_encoder', type=bool, default=False, help='Whether to use pretrained RGB encoder')
    parser.add_argument('--model_ir_pretrained_encoder', type=bool, default=False, help='Whether to use pretrained IR encoder')
    parser.add_argument('--model_freeze_encoder', type=bool, default=False, help='Freeze encoder weights or not')
    parser.add_argument('--model_verb_num_blocks', type=int, default=1, help='Number of VERB blocks')
    parser.add_argument('--model_variational', type=bool, default=True, help='Include varaitional layer or not')
    parser.add_argument('--model_sequence_length', type=int, default=1, help='Number of tokens in sequence input to joint')
    parser.add_argument('--model_mask_modality', type=bool, default=False, help='Mask out certain % of patches of one modality - only applicable for transformer based methods')
    parser.add_argument('--use_resnet50', type=bool, default=False, help='Whether to use resnet50 as encoder')
    parser.add_argument('--verb_version', type=str, default='verb', help='Version of VERB to use')
    parser.add_argument('--non_generalizable_inputs', type=bool, default=True, help='Whether to use generalizable inputs (True is do not use generalizable inputs)')

    # Optimizer & LR_scheduler parameters
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta 1 for AdamW')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta 2 for AdamW')
    parser.add_argument('--rho', type=float, default=1.0, help='Rho for Sophia optimizer')
    parser.add_argument('--lr_scheduler_name', type=str, default='one_cycle_lr', choices=['one_cycle_lr'], help='LR scheduler to use')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Max LR for one cycle LR scheduler')
    parser.add_argument('--warmup_steps', type=int, default=200, help='Warmup steps for warmup cosine LR scheduler')
    parser.add_argument('--cycles', type=float, default=0.5, help='Cycles for warmup cosine LR scheduler')

    # Training parameters
    parser.add_argument('--gpus', nargs='+', type=int, default=[1], help='List of GPUs')
    parser.add_argument('--ckpt_dir', type=str, default='/home/ubuntu/mm_public/ckpt_dir', help='Path to checkpoint directory')
    parser.add_argument('--ckpt_full_path', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--model_name', type=str, default='mm_model', help='Model name')
    parser.add_argument('--trainer_name', type=str, default='default', help='Trainer name')
    parser.add_argument('--verbose', action='store_true', help='Print more information')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Number of gradient accumulation steps')
    parser.add_argument('--max_rank', type=int, default=50, help='Max rank for validation metrics')

    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    parser.add_argument('--wandb_project', type=str, default='mm-mafia', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='default', help='Wandb run name')

    # Loss functions
    parser.add_argument('--loss_fn', type=str, default='CrossEntropy', help='Loss Function Class')
    parser.add_argument('--triplet_loss_margin', type=float, default=1.0, help='Triplet loss margin')
    parser.add_argument('--learnable_loss_weighting', type=bool, default=False, help='Learnable_loss_weighting')
    parser.add_argument('--circle_loss_gamma', type=int, default=256, help='Gamma for circle loss')
    parser.add_argument('--circle_loss_m', type=float, default=0.25, help='M for circle loss')

    # Do we have a config file to parse? 
    cfgs_config, remaining = config_parser.parse_known_args()
    if cfgs_config.config is not None:
        with open(cfgs_config.config, 'r') as f:
            config = yaml.safe_load(f)
        parser.set_defaults(**config)

    # the main arg parser parses the rest of the cfgs, the usual
    # defaults will have been overridden if config file was passed
    cfgs = parser.parse_args(remaining)

    cfgs.workers = os.cpu_count()

    return cfgs

    