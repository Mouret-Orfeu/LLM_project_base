"""
Simple evaluation utility.
Allow to evaluate a model on a dataset using the `eval_split` method.
"""

import math, random, re, string
from typing import Optional, Tuple, List

import torch
from torch.utils.data import DataLoader

from rca_llm.utils import CfgNode as CN


class Evaluator:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to evaluate on
        C.device = 'auto'
        # dataloader parameters
        C.num_workers = 4
        C.batch_size = 64

        # generation/eval defaults
        C.max_examples = 200
        C.max_new_tokens = 300
        C.do_sample = False
        C.temperature = 1.0
        C.top_k = None
        C.print_examples = 1
        return C

    def __init__(self, config, model, train_dataset, test_dataset=None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # determine the device we'll evaluate on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("evaluating on device", self.device)

    @staticmethod
    def _normalize_text(s: Optional[str]) -> str:
        # standardize the text (lowercase, removes punctuation, removes extra whitespace)
        if s is None:
            return ''
        s = s.strip().lower()
        s = s.translate(str.maketrans('', '', string.punctuation))
        s = re.sub(r'\s+', ' ', s)
        return s

    @classmethod
    def _token_lvl_f1_score(cls, prediction: str, ground_truth: str) -> float:
        pred_tokens = cls._normalize_text(prediction).split()
        gt_tokens = cls._normalize_text(ground_truth).split()
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        # count overlaps (bag-of-words)
        from collections import Counter
        pred_counts = Counter(pred_tokens)
        gt_counts = Counter(gt_tokens)
        overlap = sum((pred_counts & gt_counts).values())
        if overlap == 0:
            return 0.0
        precision = overlap / max(len(pred_tokens), 1)
        recall = overlap / max(len(gt_tokens), 1)
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def _exact_match(cls, prediction: str, ground_truth: str) -> float:
        return 1.0 if cls._normalize_text(prediction) == cls._normalize_text(ground_truth) else 0.0

    @staticmethod
    def _lcs(x: List[str], y: List[str]) -> int:
        # Dynamic programming algorithm to find the length of the Longest Common Subsequence (LCS)
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            xi = x[i]
            dpi = dp[i]
            dpi1 = dp[i + 1]
            for j in range(n):
                if xi == y[j]:
                    dpi1[j + 1] = dpi[j] + 1
                else:
                    dpi1[j + 1] = dpi1[j] if dpi1[j] >= dp[i][j + 1] else dp[i][j + 1]
        return dp[m][n]

    @classmethod
    def _rougeL_f1(cls, prediction: str, ground_truth: str) -> float:
        pred_tokens = cls._normalize_text(prediction).split()
        gt_tokens = cls._normalize_text(ground_truth).split()
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        # length of the Longest Common Subsequence (LCS)
        # (orderded well predicted tokens, not necessarily consecutive)
        lcs_len = cls._lcs(pred_tokens, gt_tokens)
        prec = lcs_len / len(pred_tokens)
        rec = lcs_len / len(gt_tokens)
        if prec + rec == 0:
            return 0.0
        return (2 * prec * rec) / (prec + rec)

    
    def eval_split(
        self,
        split: str = 'test',
        max_examples: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        print_examples: Optional[int] = None,
    ) -> Tuple[dict, list]:
        """
        Evaluate a split on:
          - perplexity over answer bytes (for labels != -100)
          - QA metrics: Exact Match (EM) and token-level F1
          - Generation metric: ROUGE-L (F1)

        Returns: (metrics_dict, examples)
          metrics_dict = { 'byte_perplexity', 'bits_per_byte', 'exact_match', 'token_lvl_f1', 'rougeL_f1', ... }
          examples = list of (question, reference_answer, generated_answer)
        """

        # resolve config fallbacks
        cfg = self.config
        max_examples = cfg.max_examples if max_examples is None else max_examples
        max_new_tokens = cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        do_sample = cfg.do_sample if do_sample is None else do_sample
        temperature = cfg.temperature if temperature is None else temperature
        top_k = cfg.top_k if top_k is None else top_k
        print_examples = cfg.print_examples if print_examples is None else print_examples

        model = self.model
        device = self.device
        if split not in {'train', 'test'}:
            raise ValueError("split must be either 'train' or 'test'")
        if split == 'train':
            dataset = self.train_dataset
        else:
            if self.test_dataset is None:
                raise ValueError("test_dataset is None but split='test' requested")
            dataset = self.test_dataset

        df = dataset.df
        # HFModelAdapter guarantees an int pad_id setup
        pad_id = int(model.hf_model.config.pad_token_id)
        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is None and hasattr(dataset, 'tokenizer'):
            tokenizer = dataset.tokenizer

        model.eval()

        # BPB (bits_per_byte) over the split
        total_nll, total_bytes = 0.0, 0
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=False,
        )
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                _, loss = model(x, y)
                tokens = (y != -100).sum().item()
                if tokens > 0 and loss is not None:
                    # loss.item() is the average negative log-likelihood per included token in the batch
                    # multiply by number of tokens to get total negative log-likelihood for this batch
                    total_nll += loss.item() * tokens

                    # count bytes in the supervised span only
                    # for each sample, take the positions where y != -100 and x != pad
                    if tokenizer is not None:
                        for i in range(x.size(0)):
                            mask = (y[i] != -100) & (x[i] != pad_id)
                            if mask.any():
                                ids = x[i][mask].tolist()
                                txt = tokenizer.decode(ids, skip_special_tokens=True)
                                total_bytes += len(txt.encode("utf-8"))

        num_bytes = max(total_bytes, 1)
        bpb = (total_nll / num_bytes) / math.log(2.0)
        byte_perplexity = 2 ** bpb

        # Generation loop for QA metrics
        total_exact_match, total_f1, total_rougeL = 0.0, 0.0, 0.0
        num_examples = min(max_examples, len(dataset))
        indices = list(range(len(dataset)))
        random.seed(3407)
        random.shuffle(indices)
        indices = indices[:num_examples]

        examples = []  # (question, reference, generated)
        with torch.no_grad():
            for i_local in indices:
                row_idx = int(dataset.ixes[i_local])
                question = str(df.loc[row_idx, 'ticket_description'])
                reference = str(df.loc[row_idx, 'ticket_resolution'])
                prompt = dataset.prompt_description_addition + question + dataset.prompt_resolution_addition

                generated = model.generate_from_prompt(
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    return_new_text_only=True,
                    skip_special_tokens=True,
                )

                examples.append((question, reference, generated))
                total_exact_match += self._exact_match(generated, reference)
                total_f1 += self._token_lvl_f1_score(generated, reference)
                total_rougeL += self._rougeL_f1(generated, reference)

        qa_em = total_exact_match / max(num_examples, 1)
        qa_f1 = total_f1 / max(num_examples, 1)
        rougeL = total_rougeL / max(num_examples, 1)

        results = {
            'split': split,
            'examples_evaluated': int(num_examples),
            'byte_perplexity': float(byte_perplexity) if total_bytes > 0 else None,
            'bits_per_byte': float(bpb) if total_bytes > 0 else None,
            'exact_match': float(qa_em),
            'token_lvl_f1': float(qa_f1),
            'rougeL_f1': float(rougeL),
        }

        if print_examples and print_examples > 0:
            for k, (q, ref, pred) in enumerate(examples[:print_examples]):
                print(f'[#{k}] QUESTION: {q}')
                print(f'     REF    : {ref}')
                print(f'     PRED   : {pred}')
                print('-' * 60)

        bpb_str = f"{results['byte_perplexity']:.3f}" if results['byte_perplexity'] is not None else "n/a"
        print(
            f"Eval {split}: byte_perplexity={bpb_str} | exact match={qa_em*100:.2f}% | token level f1={qa_f1*100:.2f}% | ROUGE-L={rougeL*100:.2f}% | examples={num_examples}"
        )
        return results, examples
