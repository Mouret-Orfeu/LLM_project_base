import torch
from torch.utils.data import Dataset
import pandas as pd

class CustomDataset(Dataset):
    """ 
    Dataset question & answers.

    Example usecase was IT incidents from itsm tickets:
    Input: "Hello IT Support, my Projector has been not displaying any signal. I've noticed ... "
    Output: "We identified the root cause as a misconfigured network setting. After patched the system, ... "

    Which will feed into the transformer as a concatenation of input and output, with added context indications at the beginning and between the two:
    example:
    enriched input: "description du ticket itsm: Hello IT Support, my Projector has been not displaying any signal. I've noticed ... "
    enriched output: "Réponse de l'équipe IT pour la résolution du ticket: We identified the root cause as a misconfigured network setting. After patched the system, ... "

    """

    # When initializing, be sure to override block_size, prompt_description_addition and prompt_resolution_addition to your liking
    def __init__(self, df, split, tokenizer, block_size = 1024, test_frac=0.2, test_cap=None, seed=None, indices=None):
        assert split in {"train", "test"}
        self.df = df.reset_index(drop=True)
        self.split = split
        self.tokenizer = tokenizer
        self.block_size = block_size

        # These 2 have to be adapted to the usecase of this dataset, or set using the setter
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
                # test_cap limits the maximum number of test samples
                num_test = min(num_test, test_cap)

            test_idx = perm[:num_test]
            train_idx = perm[num_test:]

            # the size of the dataset created
            self.ixes = test_idx if split == "test" else train_idx
    
    def set_prompt_description_addition(self, text):
        self.prompt_description_addition = text
    def set_prompt_resolution_addition(self, text):
        # test if there is "\n" at the beginning, if not, print a message warning
        if not text.startswith("\n"):
            print("Warning: You might want to add a newline character at the beginning of the prompt_resolution_addition (\\n).")
        self.prompt_resolution_addition = text

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
        question = str(self.df.loc[row_idx, 'question'])
        answer = str(self.df.loc[row_idx, 'answer'])

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

        # convert x to tensors
        # x is the full input sequence token ids fed to the model for training, 
        # from which the prompt_description_addition, question and prompt_resolution_addition tokens will be masked so that 
        # the Loss is only computed on the answer part
        x = torch.tensor(full_sequence_token_ids, dtype=torch.long)

        # y: ignore prompt tokens; learn on answer tokens
        y = x.clone()
        prompt_len = len(prompt_token_ids)  # guard if prompt >= block_size
        # Masking everything except the answer part here (when token id is -100, that token is ignored in loss computation)
        y[:prompt_len] = -100

        # right-pad to block_size if needed so that all (x,y) have same size
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        pad_id = int(pad_id)  # ensure it's an int
        pad_len = self.block_size - x.numel()
        if pad_len > 0:
            pad_x = torch.full((pad_len,), pad_id, dtype=torch.long)
            pad_y = torch.full((pad_len,), -100,  dtype=torch.long)
            x = torch.cat([x, pad_x], dim=0)  # <- tensors inside a list/tuple
            y = torch.cat([y, pad_y], dim=0)

        return x, y
