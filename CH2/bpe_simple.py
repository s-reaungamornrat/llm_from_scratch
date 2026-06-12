from collections import Counter, deque
from functools import lru_cache # caching the return values from a function

class BPETokenizerSimple:
    def __init__(self, replace_space="Ġ"):
        # maps token_id to token_str (e.g., {11246:'some'})
        self.vocab={}
        # maps token_str to token_id (e.g., {'some':11246})
        self.inversed_vocab={}
        # dict of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges={}
        # whether to replace space by this character
        self.replace_space=replace_space

    def train(self, text, vocab_size, allowed_special={'<|endoftext|>'}):
        """Train the BPE tokenizer from scratch
        Args:
            text (str): The training text
            vocab_size (int): The desired vocabulary size
            allowed_special (set): A set of special tokens to include
        """
        # Preprocess: replace spaces with "Ġ"
        # Note that "Ġ" is a particularity of the GPT-2 BPE implementation, e.g., "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # (GPT-4 BPE would tokenize it as ["Hello", " world"])
        if self.replace_space:
            processed_text=[]
            for i, char in enumerate(text):
                if char==" " and i!=0: processed_text.append(replace_space)
                if char!=" ": processed_text.append(char)
            processed_text="".join(processed_text)
        else: processed_text=text
        
        # initialize vocab with unique characters, including `replace_space` if present
        # start with the first 256 ASCII characters
        unique_chars=[chr(i) for i in range(256)]
        
        # extend unique_chars with characters from processed_text that are not already included
        unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)
        
        # optionally ensure that `replace_space` is included if it is relevant to your text processing
        if self.replace_space and self.replace_space not in unique_chars: unique_chars.append(self.replace_space)
        
        # now create the vocab and inverse vocab dicts
        self.vocab={i:char for i, char in enumerate(unique_chars)}
        self.inversed_vocab={char:i for i, char in self.vocab.items()}
        
        # add allowed_special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inversed_vocab:
                    new_id=len(self.vocab)
                    self.vocab[new_id]=token
                    self.inversed_vocab[token]=new_id
        
        # tokenize the processed_text into token IDs
        token_ids=[self.inversed_vocab[char] for char in processed_text]
        
        # repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            if len(token_ids)<2: break
            pair_id=self.find_freq_pair(token_ids, mode='most')
            if pair_id is None: break # no more pairs to merge. stop training
        
            updated=self.replace_pair(token_ids, pair_id, new_id)
            if updated==token_ids: break # cannot further replacing any token pairs
        
            token_ids=updated # len(updated)<len(token_ids)
            self.bpe_merges[pair_id]=new_id
        
            # update vocabulary immediately
            p0, p1=pair_id
            merged_token=self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id]=merged_token
            self.inversed_vocab[merged_token]=new_id

    def tokenize_with_bpe(self, token):
        """Tokenize a single token using BPE merges
        Args:
            token (str): The token to be tokenized
        Returns:
            (list[int]): The list of token IDs after applying byte-pair encoding
        """
        # tokenize the token into individual characters (as initial token IDs)
        token_ids=[self.inversed_vocab.get(char, None) for char in token]
        if None in token_ids: 
            missing_chars=[char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocabulary: {missing_chars}")
        
        can_merge=True
        while can_merge and len(token_ids)>1:
            can_merge=False
            new_tokens=[]
            i=0
            while i<len(token_ids)-1:
                pair=(token_ids[i], token_ids[i+1])
                if pair in self.bpe_merges:
                    merged_token_id=self.bpe_merges[pair]
                    new_tokens.append(merged_token_id)
                    # print(f"Merged pair {pair}-> {merged_token_id=} ('{self.vocab[merged_token_id]=}')")
                    i+=2 # skip the next token as they are merged
                    can_merge=True
                else:
                    new_tokens.append(token_ids[i])
                    i+=1
            if i<len(token_ids): new_tokens.append(token_ids[i])
            token_ids=new_tokens
        return token_ids

    def encode(self, text):
        """ Encode the input text into a list of token IDs
        Args:
            text (str): The text to be encoded
        Returns:
            (list[int]): List of token indices
        """
        tokens=[]
        # split text into tokens, keeping newlines intact
        words=text.replace("\n", " \n ").split() # ensure \n is treated as a separate token
        for i, word in enumerate(words):
            if i>0 and not word.startswith("\n"): 
                # add ' ' or replace_space to words that follow a space or newline
                tokens.append((self.replace_space if self.replace_space else ' ')+word)
            else: tokens.append(word) # handle first word or standalone '\n'
        
        token_ids=[]
        for token in tokens:
            if token in self.inversed_vocab:
                # token is contained in the vocabulary as is
                token_id=self.inversed_vocab[token]
                token_ids.append(token_id)
            else: # attempt to handle subword tokenization via BPE
                sub_token_ids=self.tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids)
        return token_ids

    def decode(self,token_ids):
        """Decode a list of token IDs back into a string
        Args:
            token_ids (list[int]): The list of token IDs to be decoded
        Returns:
            (str): Decoded string
        """
        decoded_string=""
        for token_id in token_ids:
            if token_id not in self.vocab: raise ValueError(f"Token ID {token_id} not found in vocab")
            token=self.vocab[token_id]
            if self.replace_space and token.startswith(self.replace_space): 
                # replace replace_space by space
                decoded_string+=" "+token[1:]
            else: decoded_string+=token
        return decoded_string
    
    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        """
        Args:
            token_ids (sequence[int]): Sequence of tokens extracted from text
        Returns:
            (tuple[int,int]): A pair of most/least frequent tokens
        """
        if len(token_ids)<2: return None
        pairs=Counter(zip(token_ids, token_ids[1:])) # token_ids[1:] controls the number of pairs
        if not pairs: return None
        if mode=='most': return max(pairs.items(), key=lambda x: x[1])[0] # we use [0] to get the token pair. [1] is its occurence
        elif mode=="least": return min(pairs.items(), key=lambda x: x[1])[0]
        else: raise ValueError(f"Mode {mode} is invalid. Choose between 'most' or 'least'")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        """
        Args:
            token_ids (sequence[int]): Sequence of tokens extracted from text
            pair_id (tuple[int,int]): A pair of most/least frequent token, if found in `token_ids`, will be replaced by `new_id`
            new_id (int): Token index for `pair_id`
        Returns:
            (sequence[int]): Updated sequence of tokens 
        """
        dq=deque(token_ids)
        replaced=[]
        while dq:
            current=dq.popleft()
            if dq and (current, dq[0])==pair_id: 
                replaced.append(new_id)
                # remove the 2nd token of the pair, the 1st was already removed
                dq.popleft()
            else: replaced.append(current)
        return replaced