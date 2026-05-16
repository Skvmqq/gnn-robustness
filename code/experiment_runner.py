import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import Amazon

import graph_features
from model import GCNModel, LogisticRegressionModel, MLPModel
from utils import (
    augment_edges,
    augment_features,
    feature_similarity_rewiring,
    masked_accuracy,
    set_seed,
)


def train_model_graph(
    model,
    data,
    lr=0.001,
    epochs=400,
    edge_aug_mode=None,
    edge_aug_percent=0.08,
    edge_aug_is_undirected=True,
    train_feat_aug_mode=None,
    train_feat_aug_percent=0.2,
    scores=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(epochs):
        model.train()

        if edge_aug_mode is None:
            edge_in = data.edge_index
        else:
            edge_kwargs = {
                "mode": edge_aug_mode,
                "percent": edge_aug_percent,
                "is_undirected": edge_aug_is_undirected,
            }
            if edge_aug_mode in {"bw_prob", "pagerank_prob", "degree_prob"}:
                edge_kwargs["scores"] = scores
            edge_in = augment_edges(data.edge_index, **edge_kwargs)

        if train_feat_aug_mode is None:
            train_x = data.x
        else:
            train_x = augment_features(
                data.x,
                mode=train_feat_aug_mode,
                percent=train_feat_aug_percent,
            )

        logits = model(train_x, edge_in)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}/{epochs} loss={loss.item():.4f}", flush=True)

    return model


def train_model_base(model, data, lr=0.001, epochs=400):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(epochs):
        model.train()
        prediction = model(data.x)

        loss = loss_fn(prediction[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    return model


@torch.no_grad()
def eval_under_noise(
    model,
    data,
    noise_levels,
    noise_mode="flip",
    use_shield=False,
    shield_mode="fixed",
    shield_threshold=0.15,
    shield_drop_ratio=0.1,
    shield_only_when_noisy=True,
    eval_edge_aug_mode=None,
    eval_edge_aug_percent=0.1,
    eval_edge_aug_is_undirected=True,
    eval_apply_feature_noise=True,
    eval_edge_percent_from_noise=True,
    scores=None,
):
    model.eval()
    accs = []

    for p in noise_levels:
        if eval_apply_feature_noise:
            noisy_x = augment_features(data.x, mode=noise_mode, percent=p)
        else:
            noisy_x = data.x
        apply_shield = use_shield and (not shield_only_when_noisy or p > 0)
        if apply_shield:
            eval_edge_index = feature_similarity_rewiring(
                noisy_x,
                data.edge_index,
                mode=shield_mode,
                threshold=shield_threshold,
                drop_ratio=shield_drop_ratio,
            )
        else:
            eval_edge_index = data.edge_index

        if eval_edge_aug_mode is not None:
            edge_kwargs = {
                "mode": eval_edge_aug_mode,
                "percent": p if eval_edge_percent_from_noise else eval_edge_aug_percent,
                "is_undirected": eval_edge_aug_is_undirected,
            }
            if eval_edge_aug_mode in {"bw_prob", "pagerank_prob", "degree_prob", "closeness_prob", "eigenvector_prob"}:
                edge_kwargs["scores"] = scores
            eval_edge_index = augment_edges(eval_edge_index, **edge_kwargs)

        logits = model(noisy_x, eval_edge_index)
        accs.append(masked_accuracy(logits, data.y, data.test_mask))

    return np.array(accs)


def bw_centrality(data):
    gf = graph_features.GraphFeatures(data, un_directed=True)
    scores = gf.betweenness(normalized=True)
    return torch.tensor(scores, dtype=torch.float, device=data.edge_index.device)


def pagerank_centrality(data):
    gf = graph_features.GraphFeatures(data, un_directed=True)
    pr = gf.pagerank()
    return torch.tensor(pr, dtype=torch.float, device=data.edge_index.device)


def degree_centrality(data):
    gf = graph_features.GraphFeatures(data, un_directed=True)
    dc = gf.degree_centrality()
    return torch.tensor(dc, dtype=torch.float, device=data.edge_index.device)


def closeness_centrality(data):
    gf = graph_features.GraphFeatures(data, un_directed=True)
    cc = gf.closeness_centrality()
    return torch.tensor(cc, dtype=torch.float, device=data.edge_index.device)


def eigenvector_centrality(data):
    gf = graph_features.GraphFeatures(data, un_directed=True)
    try:
        ec = gf.eigenvector_centrality()
    except Exception:
        ec = [0.0] * data.num_nodes
    return torch.tensor(ec, dtype=torch.float, device=data.edge_index.device)


def centrality_scores(data, edge_aug_mode):
    if edge_aug_mode == "bw_prob":
        return bw_centrality(data)
    if edge_aug_mode == "pagerank_prob":
        return pagerank_centrality(data)
    if edge_aug_mode == "degree_prob":
        return degree_centrality(data)
    if edge_aug_mode == "closeness_prob":
        return closeness_centrality(data)
    if edge_aug_mode == "eigenvector_prob":
        return eigenvector_centrality(data)
    return None


def _plot_results(args, gcn_results, mlp_results, lr_results):
    noise = args.noise_levels

    gcn_mean = [np.mean(x) for x in gcn_results]
    gcn_std = [np.std(x) for x in gcn_results]

    mlp_mean = [np.mean(x) for x in mlp_results]
    mlp_std = [np.std(x) for x in mlp_results]

    lr_mean = [np.mean(x) for x in lr_results]
    lr_std = [np.std(x) for x in lr_results]

    plt.figure(figsize=(8, 5))
    plt.errorbar(noise, gcn_mean, yerr=gcn_std, marker="o", capsize=5, label="GCN")
    plt.errorbar(noise, mlp_mean, yerr=mlp_std, marker="o", capsize=5, label="MLP")
    plt.errorbar(noise, lr_mean, yerr=lr_std, marker="o", capsize=5, label="LogReg")

    plt.xlabel("Noise Level")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Noise Level")
    plt.legend()
    plt.grid(True)
    plt.savefig('/kaggle/working/robustness_results.png')
    plt.show()


def run_one_dataset(dataset_name, args):
    dataset = Amazon(root='data', name=dataset_name)
    data = dataset[0]

    edge_aug_mode = getattr(args, "edge_aug_mode", "bw_prob")
    edge_aug_percent = getattr(args, "edge_aug_percent", 0.08)
    edge_aug_is_undirected = getattr(args, "edge_aug_is_undirected", True)
    train_feat_aug_mode = getattr(args, "train_feat_aug_mode", None)
    train_feat_aug_percent = getattr(args, "train_feat_aug_percent", 0.2)

    eval_noise_mode = getattr(args, "eval_noise_mode", None) or "flip"
    use_eval_shield = getattr(args, "use_eval_shield", True)
    eval_shield_mode = getattr(args, "eval_shield_mode", None) or "fixed"
    eval_shield_threshold = getattr(args, "eval_shield_threshold", 0.15)
    eval_shield_drop_ratio = getattr(args, "eval_shield_drop_ratio", 0.1)
    eval_shield_only_when_noisy = getattr(args, "eval_shield_only_when_noisy", True)
    
    eval_edge_aug_mode = getattr(args, "eval_edge_aug_mode", None)
    eval_edge_aug_percent = getattr(args, "eval_edge_aug_percent", 0.1)
    eval_edge_aug_is_undirected = getattr(args, "eval_edge_aug_is_undirected", True)
    eval_apply_feature_noise = getattr(args, "eval_apply_feature_noise", True)
    eval_edge_percent_from_noise = getattr(args, "eval_edge_percent_from_noise", True)

    scores = None

    gcn_results = [[] for _ in args.noise_levels]
    mlp_results = [[] for _ in args.noise_levels]
    lr_results = [[] for _ in args.noise_levels]

    split_seed = 0
    set_seed(split_seed)
    transform = RandomNodeSplit(
    split="test_rest",
    num_train_per_class=20,
    num_val=500,
    num_test=1000,
)
    data_run = transform(data.clone())
    scores = centrality_scores(data_run, edge_aug_mode)
    eval_scores = centrality_scores(data_run, eval_edge_aug_mode) if eval_edge_aug_mode else None
    for run in range(args.runs):
        print(f"\n--- Run {run + 1}/{args.runs} ---")
        set_seed(run)
   
        gcn = GCNModel(dataset.num_features, args, dataset.num_classes, args.dropout)
        gcn.reset_parameters()
        gcn = train_model_graph(
            gcn,
            data_run,
            args.lr,
            args.epochs,
            edge_aug_mode=edge_aug_mode,
            edge_aug_percent=edge_aug_percent,
            edge_aug_is_undirected=edge_aug_is_undirected,
            train_feat_aug_mode=train_feat_aug_mode,
            train_feat_aug_percent=train_feat_aug_percent,
            scores=scores,
        )

        mlp = MLPModel(dataset.num_features, args, dataset.num_classes, args.dropout)
        mlp.reset_parameters()
        mlp = train_model_base(mlp, data_run, args.lr, args.epochs)

        logreg = LogisticRegressionModel(max_iter=1000, random_state=run)
        logreg.fit(data_run.x[data_run.train_mask], data_run.y[data_run.train_mask])

        gcn_curve = eval_under_noise(
            gcn,
            data_run,
            args.noise_levels,
            noise_mode=eval_noise_mode,
            use_shield=use_eval_shield,
            shield_mode=eval_shield_mode,
            shield_threshold=eval_shield_threshold,
            shield_drop_ratio=eval_shield_drop_ratio,
            shield_only_when_noisy=eval_shield_only_when_noisy,
            eval_edge_aug_mode=eval_edge_aug_mode,
            eval_edge_aug_percent=eval_edge_aug_percent,
            eval_edge_aug_is_undirected=eval_edge_aug_is_undirected,
            eval_apply_feature_noise=eval_apply_feature_noise,
            eval_edge_percent_from_noise=eval_edge_percent_from_noise,
            scores=eval_scores,
        )
        mlp_curve = eval_under_noise(
            mlp,
            data_run,
            args.noise_levels,
            noise_mode=eval_noise_mode,
            eval_edge_aug_mode=eval_edge_aug_mode,
            eval_edge_aug_percent=eval_edge_aug_percent,
            eval_edge_aug_is_undirected=eval_edge_aug_is_undirected,
            eval_apply_feature_noise=eval_apply_feature_noise,
            eval_edge_percent_from_noise=eval_edge_percent_from_noise,
            scores=eval_scores,
        )

        for i, p in enumerate(args.noise_levels):
            if eval_apply_feature_noise:
                noisy_x = augment_features(data_run.x, mode=eval_noise_mode, percent=p)
            else:
                noisy_x = data_run.x
            pred_lr = logreg.predict(noisy_x[data_run.test_mask])
            pred_lr = torch.tensor(pred_lr)
            lr_acc = (pred_lr == data_run.y[data_run.test_mask]).float().mean().item()

            print(
                f"Run {run + 1}/{args.runs} | Noise={p:.1f} "
                f"GCN={gcn_curve[i]:.4f} MLP={mlp_curve[i]:.4f} LR={lr_acc:.4f}"
            )

            gcn_results[i].append(gcn_curve[i])
            mlp_results[i].append(mlp_curve[i])
            lr_results[i].append(lr_acc)

    print("\n===== FINAL (PER NOISE) =====")
    results_summary = {}
    for i, p in enumerate(args.noise_levels):
        summary = {
            "GCN": {"mean": float(np.mean(gcn_results[i])), "std": float(np.std(gcn_results[i]))},
            "MLP": {"mean": float(np.mean(mlp_results[i])), "std": float(np.std(mlp_results[i]))},
            "LogReg": {"mean": float(np.mean(lr_results[i])), "std": float(np.std(lr_results[i]))},
        }
        results_summary[f"noise_{p}"] = summary
        print(
            f"Noise {p:.1f} | "
            f"GCN {np.mean(gcn_results[i]):.4f} ± {np.std(gcn_results[i]):.4f} | "
            f"MLP {np.mean(mlp_results[i]):.4f} ± {np.std(mlp_results[i]):.4f} | "
            f"LogReg {np.mean(lr_results[i]):.4f} ± {np.std(lr_results[i]):.4f}"
        )
    
    
    with open('/kaggle/working/results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("\n Results saved to /kaggle/working/results.json")

    _plot_results(args, gcn_results, mlp_results, lr_results)
