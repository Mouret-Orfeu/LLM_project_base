"""
Trains and evaluates both models testing different training durations with cross-validation on 5 folds splits.
Saves performance results in a multi-index pandas DataFrame in ./perf/model_perf_df.pkl.
Also saves training logs and loss curves in ./perf/model_training_log/.
"""

# Parts of this code are inspired by Andrej Karpathy's minGPT: https://github.com/karpathy/minGPT


# Silence tokenizers fork warning & lower verbosity BEFORE importing transformers/tokenizers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # removes the forked-parallelism warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tokenizers")
import transformers
transformers.utils.logging.set_verbosity_error()

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm  # progress bars

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login as hf_login

from rca_llm import Trainer, Evaluator, CfgNode, set_seed, setup_logging
from rca_llm.RCADataset import RCADataset
from rca_llm.HFModelAdapter import HFModelAdapter

# Color helpers (ANSI)
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def info(msg: str):
    print(f"{GREEN}{msg}{RESET}")

def result(msg: str):
    print(f"{YELLOW}{msg}{RESET}")

# Ask for HF token via env var or input
def ask_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        token = input("Enter your Hugging Face token (it will not be saved): ").strip()
    except EOFError:
        token = ""
    return token


def create_folds(n: int, n_splits: int = 5, seed: int = 3407) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)

    # shuffled indices
    perm = rng.permutation(n)

    # separate into a list of n_splits folds of (almost) equal size (for test set split)
    return [np.asarray(idx, dtype=np.int64) for idx in np.array_split(perm, n_splits)]


def ensure_perf_store(perf_dir: Path) -> Path:
    perf_dir.mkdir(parents=True, exist_ok=True)
    return perf_dir / "model_perf_df.pkl"


def init_or_load_perf_df(perf_path: Path,
                         model_types: List[str],
                         split_names: List[str],
                         metrics: List[str],
                         max_iters: List[int]) -> pd.DataFrame:
    if perf_path.exists():
        try:
            return pd.read_pickle(perf_path)
        except Exception:
            pass
    cols = pd.MultiIndex.from_product(
        [model_types, split_names, metrics, max_iters],
        names=["model_type", "split", "perf_metric", "training_duration_max_iter"],
    )
    df = pd.DataFrame(index=["value"], columns=cols, dtype=float)
    return df


def update_perf_df(perf_df: pd.DataFrame,
                   model_type: str,
                   split_name: str,
                   max_iter: int,
                   metrics: Dict[str, float]) -> pd.DataFrame:
    for k, v in metrics.items():
        perf_df.loc["value", (model_type, split_name, k, max_iter)] = float(v) if v is not None else np.nan
    return perf_df


def recompute_split_avg(perf_df: pd.DataFrame,
                        model_types: List[str],
                        metrics: List[str],
                        max_iters: List[int],
                        split_names_no_avg: List[str],
                        split_avg_name: str = "split_avg") -> pd.DataFrame:
    for mt in model_types:
        for mi in max_iters:
            for m in metrics:
                # gather the metric values for the different splits
                # split_names_no_avg is the list of the concrete CV fold names, excluding the aggregate “split_avg”
                # ["split_1", "split_2", "split_3", "split_4", "split_5"].
                vals = [perf_df.loc["value", (mt, s, m, mi)] for s in split_names_no_avg]
                # converts to numpy array, ignoring NaNs
                arr = np.array([x for x in vals if pd.notna(x)], dtype=float)
                # compute the average, or NaN if no valid value
                avg = float(arr.mean()) if arr.size > 0 else np.nan
                # set the average value in the DataFrame
                perf_df.loc["value", (mt, split_avg_name, m, mi)] = avg
    return perf_df


# return the callback function that writes the loss to a CSV file
def write_losses_callback_factory(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        csv_path.write_text("iter,loss,iter_dt\n", encoding="utf-8")

    def _cb(trainer: Trainer):
        try:
            with csv_path.open("a", encoding="utf-8") as f:
                loss_val = float(trainer.loss.item()) if trainer.loss is not None else float("nan")
                f.write(f"{trainer.iter_num},{loss_val},{trainer.iter_dt}\n")
        except Exception:
            # never crash training on logging
            pass

    return _cb


def build_log_config(work_dir: Path,
                     model_type: str,
                     train_config: CfgNode,
                     eval_config: CfgNode,
                     data_info: Dict[str, Any]) -> CfgNode:
    cfg = CfgNode(
        system=CfgNode(work_dir=str(work_dir)),
        model=CfgNode(model_type=model_type),
        trainer=train_config,
        evaluator=eval_config,
        data=CfgNode(**data_info),
        run=CfgNode(start_time=datetime.utcnow().isoformat() + "Z")
    )
    return cfg


def main():
    set_seed(3407)

    # Ask for HF token and login
    hf_token = ask_hf_token()
    if hf_token:
        try:
            hf_login(token=hf_token)
        except Exception as e:
            info(f"Warning: could not login to Hugging Face Hub: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device: {device}")

    # Paths and constants
    data_path = Path("./data/itsm_tickets_meaningful_200_utf8.csv")
    if not data_path.exists():
        print(f"ERROR: data file not found at {data_path}")
        sys.exit(1)
    df = pd.read_csv(data_path, sep=';', encoding='utf-8')

    model_types = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B",
    ]
    max_iters_list = [200, 400, 600]
    n_splits = 5
    split_names = [f"split_{i}" for i in range(1, n_splits + 1)] + ["split_avg"]
    split_names_no_avg = [f"split_{i}" for i in range(1, n_splits + 1)]
    metrics = ["byte_perplexity", "bits_per_byte", "exact_match", "token_lvl_f1", "rougeL_f1"]

    perf_dir = Path("./perf")
    perf_path = ensure_perf_store(perf_dir)

    # Load or init the 4D performance DataFrame
    perf_df = init_or_load_perf_df(perf_path, model_types, split_names, metrics, max_iters_list)

    # precompute disjoint folds
    folds = create_folds(len(df), n_splits=n_splits, seed=3407)

    # train/eval directory
    logs_root = Path("./perf/model_training_log")
    logs_root.mkdir(parents=True, exist_ok=True)

    total_runs = len(model_types) * n_splits * len(max_iters_list)
    global_pbar = tqdm(total=total_runs, desc="Total trainings", position=0)

    # Iterate over models
    for model_type in model_types:
        info(f"\n=== Model: {model_type} ===")
        info("Loading tokenizer & model (may take a while)...")
        tokenizer = AutoTokenizer.from_pretrained(model_type, token=hf_token or True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_type,
            token=hf_token or True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = HFModelAdapter(hf_model, model_type)

        # For each split
        for fold_idx in range(n_splits):
            split_name = f"split_{fold_idx + 1}"
            info(f"\n-- {split_name} --")
            split_pbar = tqdm(total=len(max_iters_list),
                              desc=f"{model_type} | {split_name}",
                              leave=False, position=1)

            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(n_splits) if i != fold_idx])

            # Datasets for this split
            train_dataset = RCADataset(df, 'train', tokenizer, indices=train_idx)
            test_dataset = RCADataset(df, 'test', tokenizer, indices=test_idx)

            # For each training duration (max_iter)
            for max_iter in max_iters_list:
                info(f"Training with max_iter={max_iter} ...")
                train_config = Trainer.get_default_config()
                train_config.max_iters = int(max_iter)
                train_config.batch_size = 2  # small batch size for memory constraints
                train_config.num_workers = min(2, train_config.num_workers)

                trainer = Trainer(train_config, model, train_dataset)

                # Evaluator config aligned with training
                eval_config = Evaluator.get_default_config()
                eval_config.batch_size = train_config.batch_size
                eval_config.num_workers = train_config.num_workers
                eval_config.print_examples = 0

                # Prepare logging directory and files
                run_dir = logs_root / (
                    f"{model_type.replace('/', '__')}/"  # safe folder name
                    f"{split_name}/"
                    f"maxiter_{max_iter}__{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
                )
                loss_csv = run_dir / "training_loss_per_batch.csv"

                # Bookkeeping config (params, paths, split sizes)
                data_info = {
                    "data_file": str(data_path),
                    "num_rows": int(len(df)),
                    "split": split_name,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                }
                log_cfg = build_log_config(run_dir, model_type, train_config, eval_config, data_info)
                setup_logging(log_cfg)

                # Register loss logging callback
                trainer.set_callback('on_batch_end', write_losses_callback_factory(loss_csv))

                # Train
                t0 = time.time()
                trainer.run()
                t1 = time.time()
                train_seconds = t1 - t0

                try:
                    (run_dir / "train_summary.json").write_text(
                        json.dumps({
                            "model_type": model_type,
                            "split": split_name,
                            "max_iter": int(max_iter),
                            "train_seconds": train_seconds,
                        }, indent=2),
                        encoding="utf-8"
                    )
                except Exception:
                    pass

                # Evaluate (test split)
                info("Evaluating...")
                evaluator = Evaluator(eval_config, model, train_dataset, test_dataset)
                test_metrics, _ = evaluator.eval_split('test')

                # Print results in yellow
                result(
                    f"Eval {split_name}: "
                    f"byte_perplexity={test_metrics.get('byte_perplexity'):.3f} | "
                    f"bits_per_byte={test_metrics.get('bits_per_byte'):.3f} | "
                    f"exact match={test_metrics.get('exact_match'):.2f}% | "
                    f"token level f1={test_metrics.get('token_lvl_f1'):.2f}% | "
                    f"ROUGE-L={test_metrics.get('rougeL_f1'):.2f}%"
                )

                # Update the performance store
                perf_df = update_perf_df(perf_df, model_type, split_name, int(max_iter), {
                    "byte_perplexity": test_metrics.get("byte_perplexity"),
                    "bits_per_byte": test_metrics.get("bits_per_byte"),
                    "exact_match": test_metrics.get("exact_match"),
                    "token_lvl_f1": test_metrics.get("token_lvl_f1"),
                    "rougeL_f1": test_metrics.get("rougeL_f1"),
                })

                # Recompute split_avg with current data
                perf_df = recompute_split_avg(perf_df, [model_type], metrics, [int(max_iter)], split_names_no_avg)
                perf_df.to_pickle(perf_path)

                # Progress bars
                split_pbar.update(1)
                global_pbar.update(1)

            split_pbar.close()

    global_pbar.close()
    info(f"\nSaved performance dataframe to: {perf_path}")


if __name__ == "__main__":
    main()
