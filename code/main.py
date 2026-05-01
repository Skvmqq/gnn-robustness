from experiment_runner import run_one_dataset

class Args:
    hidden = 64
    dropout = 0.5
    lr = 0.001
    epochs = 400
    runs = 10
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.9]
    edge_aug_mode = None
    edge_aug_percent = 0.07
    edge_aug_is_undirected = True
    train_feat_aug_mode = None
    train_feat_aug_percent = 0.1
    eval_noise_mode = "noise"
    use_eval_shield = False
    eval_shield_mode = "fixed"
    eval_shield_threshold = 0.15
    eval_shield_drop_ratio = 0.10
    eval_shield_only_when_noisy = True

def main():
    args = Args()
    for name in ["CS"]:
        run_one_dataset(name, args)


if __name__ == "__main__":
    main()