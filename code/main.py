from experiment_runner import run_one_dataset
import torch

class BaseArgs:
    hidden = 64
    dropout = 0.5
    lr = 0.001
    epochs = 400
    runs = 5 # Reduced from 10 for faster Kaggle execution
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.9]
    eval_noise_mode = "noise"
    use_eval_shield = False
    eval_shield_mode = "fixed"
    eval_shield_threshold = 0.15
    eval_shield_drop_ratio = 0.10
    eval_shield_only_when_noisy = True

def run_suite():
    
    scenarios = [
        {"name":"TrainEdge_None__Eval_FeatureNoiseOnly","feat_mode":None,"edge_mode":None, "eval_edge_mode": None, "eval_feature_noise": True},
        {"name":"TrainEdge_None__Eval_EdgeDropoutOnly","feat_mode":None,"edge_mode":None, "eval_edge_mode": "dropout", "eval_feature_noise": False},
        {"name":"TrainEdge_None__Eval_FeatureNoisePlusEdgeDropout","feat_mode":None,"edge_mode":None, "eval_edge_mode": "dropout", "eval_feature_noise": True},
        
        {"name":"TrainEdge_BWProb__Eval_FeatureNoiseOnly","feat_mode":None,"edge_mode":"bw_prob", "eval_edge_mode": None, "eval_feature_noise": True},
        {"name":"TrainEdge_BWProb__Eval_EdgeDropoutOnly","feat_mode":None,"edge_mode":"bw_prob", "eval_edge_mode": "dropout", "eval_feature_noise": False},
        {"name":"TrainEdge_BWProb__Eval_FeatureNoisePlusEdgeDropout","feat_mode":None,"edge_mode":"bw_prob", "eval_edge_mode": "dropout", "eval_feature_noise": True},
        
        {"name":"TrainEdge_PageRankProb__Eval_FeatureNoiseOnly","feat_mode":None,"edge_mode":"pagerank_prob", "eval_edge_mode": None, "eval_feature_noise": True},
        {"name":"TrainEdge_PageRankProb__Eval_EdgeDropoutOnly","feat_mode":None,"edge_mode":"pagerank_prob", "eval_edge_mode": "dropout", "eval_feature_noise": False},
        {"name":"TrainEdge_PageRankProb__Eval_FeatureNoisePlusEdgeDropout","feat_mode":None,"edge_mode":"pagerank_prob", "eval_edge_mode": "dropout", "eval_feature_noise": True},
        
        {"name":"TrainEdge_DegreeProb__Eval_FeatureNoiseOnly","feat_mode":None,"edge_mode":"degree_prob", "eval_edge_mode": None, "eval_feature_noise": True},
        {"name":"TrainEdge_DegreeProb__Eval_EdgeDropoutOnly","feat_mode":None,"edge_mode":"degree_prob", "eval_edge_mode": "dropout", "eval_feature_noise": False},
        {"name":"TrainEdge_DegreeProb__Eval_FeatureNoisePlusEdgeDropout","feat_mode":None,"edge_mode":"degree_prob", "eval_edge_mode": "dropout", "eval_feature_noise": True},
        
    ]

    for scenario in scenarios:
        print(f"\n{'#'*30}")
        print(f"STARTING SCENARIO: {scenario['name']}")
        print(f"{'#'*30}\n")
        
       
        args = BaseArgs()
        args.train_feat_aug_mode = scenario['feat_mode']
        args.train_feat_aug_percent = 0.1
        args.edge_aug_mode = scenario['edge_mode']
        args.edge_aug_percent = 0.1
        args.edge_aug_is_undirected = True
        args.eval_edge_aug_mode = scenario['eval_edge_mode']
        args.eval_edge_aug_percent = 0.1
        args.eval_edge_aug_is_undirected = True
        args.eval_apply_feature_noise = scenario.get('eval_feature_noise', True)
        args.eval_edge_percent_from_noise = True

       
        try:
            run_one_dataset("CS", args)
            print(f"FINISHED SCENARIO: {scenario['name']}")
        except Exception as e:
            print(f"FAILED SCENARIO {scenario['name']}: {str(e)}")
        
        # Clear GPU cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_suite()