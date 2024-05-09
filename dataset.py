import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class TweetDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.tweets = []
        self.targets = []
        self.location = []
        self.padding_length = 282  # Max length of a tweet plus BOS and EOS tokens
        self.vocab_size = 0
        self.char_to_int = {}
        self.int_to_char = {}
        self.load_dataset() # Load dataset and set tweets and targets
        self.tweet_token_dict(' '.join(self.tweets)) # Prepare the token dictionaries

    def load_dataset(self):
        try:
            self.df = pd.read_csv(self.dataset_path)
            self.location = self.df['location'].tolist()
            self.tweets = self.df['text'].tolist()
            self.targets = self.df['target'].tolist()
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            self.tweets = []

    def tweet_token_dict(self, tweet_strings):
        chars = sorted(set(tweet_strings))
        self.char_to_int = {char: i for i, char in enumerate(chars)}
        self.int_to_char = {i: char for i, char in enumerate(chars)}
        self.initialize_special_tokens()
        self.vocab_size = len(self.char_to_int)

    def initialize_special_tokens(self):
        start_index = max(self.char_to_int.values(), default=0) + 1
        self.char_to_int['<BOS>'] = start_index
        self.char_to_int['<EOS>'] = start_index + 1
        self.char_to_int['<PAD>'] = start_index + 2
        self.char_to_int['<UNK>'] = start_index + 3
        self.int_to_char[start_index] = '<BOS>'
        self.int_to_char[start_index + 1] = '<EOS>'
        self.int_to_char[start_index + 2] = '<PAD>'
        self.int_to_char[start_index + 3] = '<UNK>'

    def tokenize_tweet(self, tweet):
        base_tokens = [self.char_to_int['<BOS>']] + [self.char_to_int.get(char, self.char_to_int['<UNK>']) for char in tweet] + [self.char_to_int['<EOS>']]
        if len(base_tokens) < self.padding_length:
            base_tokens += [self.char_to_int['<PAD>']] * (self.padding_length - len(base_tokens))
        return base_tokens[:self.padding_length]

    def detokenize_tweet(self, tokenized_tweet):
        return ''.join([self.int_to_char[token] for token in tokenized_tweet if token in self.int_to_char])

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        target = self.targets[idx]
        tokenized_tweet = self.tokenize_tweet(tweet)
        return (tokenized_tweet, target)

# Assuming your TweetDataset is defined and loaded as 'dataset'
def collate_fn(batch):
    tweets, targets = zip(*batch)
    return torch.tensor(tweets), torch.tensor(targets)

"""
print(len(dataset))
print(f"Tweet: {dataset.tweets[1]}")
print(f"Target: {dataset.targets[1]}")
print(f"Vocab size: {dataset.vocab_size}")
print(f"Location: {dataset.location[1]}")
print(f"char_to_int: {dataset.char_to_int}")
print(f"Tokenized tweet: {dataset[1][0]}")
print(f"Target: {dataset[1][1]}")
print(f"Detokenized tweet: {dataset.detokenize_tweet(dataset[1][0])}")


path = "./data/train.csv"
dataset = TweetDataset(path)



# Create the DataLoader instance
batch_size = 8  
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

"""
