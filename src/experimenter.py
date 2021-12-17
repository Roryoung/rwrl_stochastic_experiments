import os
import json
import argparse
import importlib

from trainer import Trainer 
from evaluator import Evaluator
from utils import load_pkl_file


class Experimenter():
    def __init__(self, manifest_loc):
        if not os.path.exists(manifest_loc):
            raise ImportError(f"Unable to import from: `{manifest_loc}` ")

        spec = importlib.util.spec_from_file_location(manifest_loc, manifest_loc)
        manifest_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manifest_module)

        self.manifest = manifest_module.get_manifest()


    def train(self):
        for trial in self.manifest:
            exp_dir = f"results/{trial['exp_name']}/{trial['trial_name']}/{trial['model_name']}"

            if not os.path.exists(f"{exp_dir}/seed_info.pkl"):
                trial["training_mode"] = "collect"
            else:
                if trial["training_mode"] == "skip_existing":
                    print(f"Skippig {trial['exp_name']} | {trial['trial_name']} | {trial['model_name']}")
                    continue

            if trial["training_mode"] == "extend":
                seed_info = load_pkl_file(f"{exp_dir}/seed_info.pkl")
                n_seeds = seed_info["total_trials"] + 1
            else:
                n_seeds = trial["n_seeds"]

            
            for i in range(n_seeds):
                print(f"{trial['exp_name']} | {trial['trial_name']} | {trial['model_name']} | trial={i}")

                trainer = Trainer(exp_dir=exp_dir, trial_no=i, **trial)
                trainer.learn(**trial["learn"])
                trainer.evaluate(**trial["eval_seed"])
                trainer.close()

            evaluator = Evaluator(exp_dir=exp_dir, **trial)
            evaluator.evaluate()
            evaluator.close()


    def evaluate(self, model_loc=None):
        for trial in self.manifest:
            exp_dir = f"results/{trial['exp_name']}/{trial['trial_name']}/{trial['model_name']}"

            if os.path.exists(f"{exp_dir}/seed_info.pkl"):
                seed_info = load_pkl_file(f"{exp_dir}/seed_info.pkl")
                n_seeds = seed_info["total_trials"] + 1

                trial["training_mode"] = "extend"

                for i in range(n_seeds):
                    print(f"{trial['exp_name']} | {trial['trial_name']} | trial={i}")
                    
                    if os.path.exists(exp_dir):
                        trainer = Trainer(exp_dir=exp_dir, trial_no=i, **trial)
                        
                        if model_loc is not None:
                            trainer.load(model_loc)
                        else:
                            trainer.load(f"trial_{i}/final_model.zip")

                        trainer.evaluate(**trial["eval_seed"])
                        trainer.close()
                    else:
                        print("There is no training data for this trial.")
                
                evaluator = Evaluator(exp_dir=exp_dir, **trial)
                evaluator.evaluate()
                evaluator.close()

    
    def collect_results(self):
        all_results = {}
        
        for trial in self.manifest:
            exp_dir = f"results/{trial['exp_name']}/{trial['trial_name']}/{trial['model_name']}"

            if trial['trial_name'] not in all_results:
                all_results[trial['trial_name']] = {}

            if not os.path.exists(f"{exp_dir}/evaluation/summary/metrics.json"):
                all_results[trial['trial_name']][trial["model_name"]] = None
            else:
                with open(f"{exp_dir}/evaluation/summary/metrics.json", "r") as f:
                    try:
                        saved_metrics = json.load(f)
                    except:
                        saved_metrics = None

                    all_results[trial['trial_name']][trial["model_name"]] = saved_metrics
            

        with open(f"results/{trial['exp_name']}/metrics.json", "w+") as f:
            json.dump(all_results, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="The location of the manifest to experiment")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "eval", "collect"], help='Determines if the model is trained or evaulated')
    parser.add_argument("--model_loc", type=str, default=None, help="Location of backup model to load")
    args = parser.parse_args()

    experimenter = Experimenter(args.manifest)

    if args.mode =="train":
        experimenter.train()
    elif args.mode == "eval":
        experimenter.evaluate(model_loc=args.model_loc)
    elif args.mode == "collect":
        pass
    else:
        raise Exception("Unknown value of mode. This musst be either `train` or `eval`.")

    experimenter.collect_results()
    


