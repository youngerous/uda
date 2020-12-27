import argparse


def load_config():
    parser = argparse.ArgumentParser()

    # default setting
    parser.add_argument("--train", action="store_true", help="do train")
    parser.add_argument("--test", action="store_true", help="do test")
    parser.add_argument(
        "--path", type=str, default="./data/cifar-10-batches-py/", help="Root path of CIFAR10"
    )
    parser.add_argument("--n_gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--eval_step", type=int, default=10)

    # uda setting
    parser.add_argument(
        "--do_uda",
        action="store_true",
        help="Whether to run unsupervised Data Augmenation with Consistency training.",
    )
    parser.add_argument(
        "--num_labeled",
        type=int,
        default=40000,
        help="Number of labeled images to use in UDA (rest of images become unlabeled)",
    )
    parser.add_argument("--batch_size_l", type=int, default=64, help="Labeled batch size")
    parser.add_argument("--batch_size_u", type=int, default=224, help="Unlabeled batch size")
    parser.add_argument("--batch_size_val", type=int, default=128, help="Evaluation batch size")
    parser.add_argument(
        "--lambda",
        type=float,
        default=1.0,
        help="The coefficient on the UDA loss. When you have extermely few samples, consider increasing this value",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sharpening unlabeled predictions",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Threshold for unlabeled confidence based masking",
    )
    parser.add_argument("--tsa", type=str, default="linear", help="tsa scheduling(linear/exp/log)")
    parser.add_argument(
        "--tsa_max_step",
        type=int,
        default=2000,
        help="Max step for calaulating tsa schedule. Originally it should be max step of your training, but fixed in this implementation.",
    )

    args = parser.parse_args()
    return args