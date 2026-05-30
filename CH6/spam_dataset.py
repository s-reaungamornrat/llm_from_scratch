import requests
import zipfile
import os
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    """Identify the longest sequence in the training dataset and add the padding token to the others to match that sequence length"""
    
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        
        self.data=pd.read_csv(csv_file)
        
        # pre-tokenize texts
        self.encoded_texts=[tokenizer.encode(text) for text in self.data['Text']]

        if max_length is None: self.max_length=self._longest_encoded_length()
        else:
            self.max_length=max_length
            # truncate sequences if they are longer than max_length
            self.encoded_texts=[encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        # pad sequences to the longest sequence
        self.encoded_texts=[encoded_text+[pad_token_id]*(self.max_length-len(encoded_text)) for encoded_text in self.encoded_texts]

    def _longest_encoded_length(self): return max(len(encoded_text) for encoded_text in self.encoded_texts)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index to data
        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Token index tensor of type int64 and label tensor of type int64
        """
        encoded=self.encoded_texts[idx]
        label=self.data.iloc[idx]["Label"]
        return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """
    Args:
        data_file_path (Path): File path of data
        
    """
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction")
        return

    # downloading the file
    response=requests.get(url, stream=True, timeout=60, verify=False)
    response.raise_for_status()
    with open(zip_path, 'wb') as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk: out_file.write(chunk)

    # unzipping file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(extracted_path)

    # add .tsv file extension
    original_file_path=Path(extracted_path)/"SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

def create_balanced_dataset(df):
    # count the instances of "spam"
    num_spam=df[df["Label"]=="spam"].shape[0]

    # randomly sample "ham" instances to match the number of "spam" instances
    ham_subset=df[df["Label"]=="ham"].sample(num_spam, random_state=123)

    # combine "ham" subset with "spam"
    balanced_df=pd.concat([ham_subset, df[df["Label"]=="spam"]])
    return balanced_df

def random_split(df, train_frac, validation_frac):
    # shuffle the entire dataframe
    df=df.sample(frac=1, random_state=123).reset_index(drop=True)

    # calculate split indices
    train_end=int(len(df)*train_frac)
    validation_end=train_end+int(len(df)*validation_frac)

    # split data
    train_df=df[:train_end]
    validation_df=df[train_end:validation_end]
    test_df=df[validation_end:]
    return train_df, validation_df, test_df
    