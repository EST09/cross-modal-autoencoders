import argparse

def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir", default="/Users/esthomas/Andor_Rotation/github_repo/cross-modal-autoencoders/my_save_dir")
    options.add_argument('--save-freq', action="store", dest="save_freq", default=20, type=int)
    options.add_argument('--pretrained-file', action="store", default="/Users/esthomas/Andor_Rotation/github_repo/cross-modal-autoencoders/my_save_dir/models/1.pth") #assuming this is the decoder?

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=4, type=int)
    options.add_argument('-w', '--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-4, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-4, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=2, type=int)
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float)
    options.add_argument('--train-imagenet', action="store_true")
    options.add_argument('--conditional', action="store_true")
    options.add_argument('--conditional-adv', action="store_true")

    # hyperparameters
    options.add_argument('--alpha', action="store", default=0.1, type=float)
    options.add_argument('--beta', action="store", default=1., type=float)
    options.add_argument('--lamb', action="store", default=0.00000001, type=float)
    options.add_argument('--latent-dims', action="store", default=128, type=int)

    # gpu options
    options.add_argument('-gpu', '--use-gpu', action="store_false", dest="use_gpu")

    args, unknown = options.parse_known_args()

    return args