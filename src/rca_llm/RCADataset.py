import torch
from torch.utils.data import Dataset
import pandas as pd

class RCADataset(Dataset):
    """ 
    Dataset question & answers about IT incidents from itsm tickets.
    Input: Bonjour l'équipe IT, mon projecteur pose problème. Il n'affiche aucun signal chaque fois que ... 
    Output: Merci pour votre signalement. Le problème était lié à une panne serveur...

    Which will feed into the transformer as a concatenation of input and output, with added context indications at the begining and between the two:
    example:
    description du ticket itsm: Bonjour l'équipe IT, mon projecteur pose problème. Il n'affiche aucun signal chaque fois que ... 
    Réponse de l'équipe IT pour la résolution du ticket: Merci pour votre signalement. Le problème était lié à une panne serveur...

    """

    def __init__(self, df, split, tokenizer, block_size = 1024, test_frac=0.2, test_cap=None, seed=None, indices=None):
        assert split in {"train", "test"}
        self.df = df.reset_index(drop=True)
        self.split = split
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.prompt_description_addition = "description du ticket itsm: "
        self.prompt_resolution_addition = "\nRéponse de l'équipe IT pour la résolution du ticket: "
        

        # Build split indices deterministically if a seed is provided, or use
        # explicit indices if given. This ensures train/test are disjoint when
        # both datasets are constructed with the same seed or shared indices.
        N = len(self.df)
        if indices is not None:
            # indices should be a 1D tensor/list of row indices for this split
            self.ixes = torch.as_tensor(indices, dtype=torch.long)
        else:
            if seed is not None:
                g = torch.Generator()
                g.manual_seed(int(seed))
                perm = torch.randperm(N, generator=g)
            else:
                perm = torch.randperm(N)

            num_test = int(N * test_frac)
            if test_cap is not None:
                num_test = min(num_test, test_cap)

            test_idx = perm[:num_test]
            train_idx = perm[num_test:]

            self.ixes = test_idx if split == "test" else train_idx

    def __len__(self):
        return self.ixes.numel()
    
    # getter for dataset length
    def get_length(self):
        return self.__len__()

    def get_block_size(self):
        # -1 because the last token does not ever plug back for prediction
        return self.block_size - 1 

    # get single couple (x,y) for training with the dataloader
    # tokenize, concatenate, truncate, pad to block_size, return tensors
    def __getitem__(self, i):

        row_idx = int(self.ixes[i])
        question = str(self.df.loc[row_idx, 'ticket_description'])
        answer = str(self.df.loc[row_idx, 'ticket_resolution'])

        # prompt/answer texts
        prompt = self.prompt_description_addition + question + self.prompt_resolution_addition
        # enforce EOS at the end of answer if available
        # or "" is just a fallback option if the tokenizer has no eos_token (should not happen given the test in HFmodelAdapter builder)
        eos = self.tokenizer.eos_token or ""
        answer = answer + eos

        # tokenize without auto special tokens so we fully control sequence
        encoded_prompt = self.tokenizer(prompt, add_special_tokens=False)
        encoded_answer = self.tokenizer(answer, add_special_tokens=False)

        prompt_token_ids = encoded_prompt["input_ids"]
        answer_token_ids = encoded_answer["input_ids"]

        # concatenate
        full_sequence_token_ids = prompt_token_ids + answer_token_ids

        # if prompt alone overflow block size, truncate to block_size: keep as much prompt as fits, drop answer
        if len(prompt_token_ids) >= self.block_size:
            print(f"Warning: prompt length {len(prompt_token_ids)} >= block_size {self.block_size}. Truncating prompt.")
            prompt_token_ids = prompt_token_ids[:self.block_size]
            full_sequence_token_ids = prompt_token_ids

        # if full sequence overflow block size, truncate to block_size: keep full prompt, then as much answer as fits
        if len(full_sequence_token_ids) > self.block_size:
            print(f"Warning: full sequence length {len(full_sequence_token_ids)} > block_size {self.block_size}. Truncating answer.")
            free_contextual_window_space = max(self.block_size - len(prompt_token_ids), 0)
            answer_token_ids = answer_token_ids[:free_contextual_window_space]
            full_sequence_token_ids = prompt_token_ids + answer_token_ids

        # convert to tensors
        x = torch.tensor(full_sequence_token_ids, dtype=torch.long)

        # y: ignore prompt tokens; learn on answer tokens
        y = x.clone()
        prompt_len = len(prompt_token_ids)  # guard if prompt >= block_size
        y[:prompt_len] = -100

        # right-pad to block_size if needed
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        pad_id = int(pad_id)  # ensure it's an int
        pad_len = self.block_size - x.numel()
        if pad_len > 0:
            pad_x = torch.full((pad_len,), pad_id, dtype=torch.long)
            pad_y = torch.full((pad_len,), -100,  dtype=torch.long)
            x = torch.cat([x, pad_x], dim=0)  # <- tensors inside a list/tuple
            y = torch.cat([y, pad_y], dim=0)

        return x, y
