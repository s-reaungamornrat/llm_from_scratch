from collections import Counter, deque
from functools import lru_cache
import re
import json

class BPETokenizerSimple:

    def __init__(self, replace_space=None):
        # map token_id to token_str (e.g., {11246:"some"})
        self.vocab={}
        # map token_str to token_id (e.g., {'some':11246})
        self.inverse_vocab={}
        
        # dict of BPE merges: {(token_id1, token_id2):merged_token_id} used when we do not use `bpe_rank` or not use openai
        self.bpe_merges={}

        # for the official openai GPT-2 merges, use a rank dict of form {(string_A, string_B): rank}, where
        # lower rank = higher priority
        self.bpe_ranks={}
        
        self.replace_space=replace_space
        

    @staticmethod
    def pretokenize_text(text, replace_space):
        """Break text into a list of tokens and add a space as prefix/suffix if the corresponding word begins with space characters
        or end with space characters (e.g., space, tab)
        Args:
            text (str): Input text
            replace_space (str|None): Character used to replace space
        Returns:
            (list[str]): Sequence of tokens
        """
        tokens=[]
        parts=re.split(r"(\r\n|\r|\n)", text) # split by \r\n or \r or \n
        for part in parts:
            if part=="": continue
            if part=="\r\n": 
                tokens.append("\r"); tokens.append("\n")
                continue
            if part=='\r': tokens.append('\r'); continue
            if part=='\n': tokens.append('\n'); continue
        
            # normal chunk without line breaks:
            # -If spaces precede a word, prefix the first word with `replace_space` and add standalone `replace_space` for additional spaces
            # -If spaces trail the chunk (e.g., before a newline) add standalone `replace_space` tokens 
            pending_spaces=0
            # `|` means match this or that
            # ( +) means matches one or more consecutive space characters (like spaces, tabs, etc., i.e., matches literal spaces here)
            # (\S+) means matches one or more consecutive non-whitespace characters (letters, numbers, punctuation).
            for m in re.finditer(r'( +)|(\S+)', part):
                if m.group(1) is not None: # from the above regular expression, group 1 is space, i.e., ( +)
                    pending_spaces+=len(m.group(1))
                else: # from the above regular expression, group 2 is non-space
                    word=m.group(2)
                    if pending_spaces>0: # if spaces precede words
                        for _ in range(pending_spaces-1): tokens.append(replace_space if replace_space else " ") # remaining spaces as standalone
                        tokens.append((replace_space if replace_space else " ")+word) # one leading space
                        pending_spaces=0
                    else: tokens.append(word)
            # trailing spaces (no following word): add standalone replace_space tokens
            for _ in range(pending_spaces): tokens.append(replace_space if replace_space else " ")
                
        return tokens 

    @staticmethod
    def find_freq_pair(token_id_sequences, mode='most'):
        """
        Args:
            token_id_sequences (list[list[int]]): List of lists of token IDs, where each token ID list represents a word/token/character
        Returns:
            (tuple[int,int]): A pair of token IDs that either appears most/least frequence
        """
        pairs=Counter(pair for token_ids in token_id_sequences for pair in zip(token_ids, token_ids[1:]))
        if not pairs: return None
        if mode=='most': return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode=='least': return min(pairs.items(), key=lambda x: x[1])[0]
        else: raise ValueError(f"Mode {mode} is not supported. Choose 'most' or 'least'")

    @staticmethod
    def replace_pair(token_id_sequences, pair_id, new_id):
        """
        Args:
            token_id_sequences (list[list[int]]): List of lists of token IDs, where each token ID list represents a word/token/character
        Returns:
            (list[list[int]]): List of lists of token IDs, where each token ID list represents a word/token/character
        """
        replaced_sequences=[]
        
        for token_ids in token_id_sequences: 
            dq=deque(token_ids)
            replaced=[]
        
            while dq:
                current=dq.popleft()
                if dq and (current, dq[0])==pair_id:
                    replaced.append(new_id)
                    # remove the 2nd token of the pair, the 1st was already removed
                    dq.popleft()
                else: replaced.append(current)
            replaced_sequences.append(replaced)
        return replaced_sequences

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        """Train the BPE tokenizer from scratch. Here we build vocabulary and its inverse
        Args:
            text (str): The training text
            vocab_size (int): The desired vocabulary size
            allowed_special (set): A set of special tokens to include
        """
        # pre-tokenize training text using the same boundary rules as encode()
        tokens=self.pretokenize_text(text, replace_space=self.replace_space) # list of tokens
        
        # initialize vocab with unique characters, including replace_space if present
        # start with the first 256 ASCII characters
        unique_chars=[chr(i) for i in range(256)]
        unique_chars.extend(
            char for char in sorted({char for token in tokens for char in token})
        )
        if self.replace_space and self.replace_space not in unique_chars: unique_chars.append(self.replace_space)
        
        self.vocab={i:char for i, char in enumerate(unique_chars)}
        self.inverse_vocab={char:i for i, char in self.vocab.items()}
        
        # add allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id=len(self.vocab)
                    self.vocab[new_id]=token
                    self.inverse_vocab[token]=new_id
        
        # tokenize each pre-token into character IDs
        token_id_sequences=[[self.inverse_vocab[char] for char in token] for token in tokens]
        
        # repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id=self.find_freq_pair(token_id_sequences, mode='most')
            if pair_id is None: break
            token_id_sequences=self.replace_pair(token_id_sequences, pair_id, new_id)
            self.bpe_merges[pair_id]=new_id
            
            # update vocabulary immediately
            p0, p1=pair_id
            merged_token=self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id]=merged_token
            self.inverse_vocab[merged_token]=new_id

    def load_vocab_and_merge_from_openai(self, vocab_path, bpe_merges_path):
        """ Load pre-trained vocabulary and BPE merges from openai GPT-2 files. We assume no training of tokenizer was performed prior to loading
        since this loading will overwrite all training.
        Args:
            vocab_path (str): Path to the vocab file (GPT-2 calls it 'encoder.json')
            bpe_merges_path (str): Path to the bpe_merges file (GPT-2 calls it 'vocab.bpe').
        """
        self.replace_space="Ġ" # if load from openai GPT-2, force use of replace_space="Ġ"
        # load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as file:
            loaded_vocab=json.load(file)
            # encoder.json is {token_str:id}; we want id->str and str->id
            self.vocab={int(v):k for k, v in loaded_vocab.items()}
            self.inverse_vocab={k:int(v) for k, v in loaded_vocab.items()}
        
        # must have GPT-2's printable newline character "Ċ" (U+010A) at ID 198
        if "Ċ" not in self.inverse_vocab or self.inverse_vocab["Ċ"]!=198:
            raise KeyError("Vocabulary missing GPT-2 newline glyph 'Ċ' at ID 198")
        
        # must have <|endoftext|> at 50256
        if "<|endoftext|>" not in self.inverse_vocab or self.inverse_vocab["<|endoftext|>"]!=50256:
            raise KeyError("Vocabulary missing <|endoftext|> at ID 50256")
        
        # provide a convenience aloas for '\n' -> 198
        # keep printable character 'Ċ' in vocab so BPE merges keep working
        if "\n" not in self.inverse_vocab: self.inverse_vocab["\n"]=self.inverse_vocab['Ċ']
        if "\r" not in self.inverse_vocab:
            if 201 in self.vocab: self.inverse_vocab["\r"]=201
            else: raise KeyError("Vocabulary missing carriage return token at ID 201")
        
        # load GPT-2 merges and store rank
        self.bpe_ranks={}
        with open(bpe_merges_path, 'r', encoding='utf-8') as file:
            lines=file.readlines()
            if lines and lines[0].startswith("#"): lines=lines[1:] # `#` is comment
        
            rank=0
            for line in lines:
                # * is `for extended iterable unpacking`, used to prevent the code from crashing when we don't know the exact number of subwords 
                # or characters a tokenizer is going to return, while ensuring that we successfully capture the very first token
                token1, *rest=line.strip().split() 
                # BPE works by merging frequent pairs of characters. We consider noise if multiple characters merged together so we ignore
                if len(rest)!=1: continue 
                token2=rest[0]
                if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                    self.bpe_ranks[(token1, token2)]=rank
                    rank+=1
                else: pass # safe to skip pairs whose symbols are not in vocab

    def tokenize_with_bpe(self, token):
        """Tokenize a single token using BPE merges
        Args:
            token (str): The token to tokenize
        Returns:
            (list[int]): List of token IDs after appling BPE
        """
        # tokenize the token into individual characters (as initial token IDs)
        token_ids=[self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids: 
            missing_chars=[char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocabulary: {missing_chars}")
        
        # if we have not loaded openai's GPT-2 merges, do the following
        if not self.bpe_ranks:
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
                        i+=2 # skip the next token as it's merged
                        can_merge=True
                    else: new_tokens.append(token_ids[i]); i+=1
                if i<len(token_ids): new_tokens.append(token_ids[i])
                token_ids=new_tokens
            return token_ids
        
        # otherwise, do GPT-2 style mergeing with the ranks:
        # convert token_ids back to string "symbols" for each ID
        # e.g., token='Arrhythmogenic'
        # token_ids=[32, 81, 81, 71, 88, 83, 71, 76, 78, 70, 68, 77, 72, 66]->
        # symbols=['A', 'r', 'r', 'h', 'y', 't', 'h', 'm', 'o', 'g', 'e', 'n', 'i', 'c']
        symbols=[self.vocab[id_num] for id_num in token_ids]
        
        # repeated merge all occurrences of the lowest-rank pair
        while True:
            # collect all adjacent pairs
            pairs=set(zip(symbols, symbols[1:])) # e.g., {('A', 'r'),('r', 'r'),('r', 'h'),..}
            if not pairs: break
        
            # find the pair with the best (lowest) rank
            min_rank=float('inf')
            bigram=None
            for p in pairs:
                r=self.bpe_ranks.get(p, float('inf'))
                if r<min_rank: min_rank=r; bigram=p
            # if no valid ranked pair is present, we are done
            if bigram is None or bigram not in self.bpe_ranks: break
        
            # merge all occurrences of that pair
            first, second=bigram
            new_symbols=[]
            i=0
            while i<len(symbols):
                # if we see (first, second) at position i, merge them
                if i<len(symbols)-1 and symbols[i]==first and symbols[i+1]==second:
                    new_symbols.append(first+second) # merged symbol, e.g., first='e', second='n', (first+second)='en'
                    i+=2
                else:
                    new_symbols.append(symbols[i])
                    i+=1
            symbols=new_symbols
            if len(symbols)==1: break
                
        # convert merged symbols back to id, e.g., symbols=['Ar', 'rh', 'ythm', 'ogenic']->[3163, 17179, 34853, 15147]
        merged_ids=[self.inverse_vocab[sym] for sym in symbols]
        return merged_ids

    def encode(self, text, allowed_special=None):
        """Encode the input text into a list of token IDs, with tiktoken-style handling of special tokens
        Args:
            text (str): The input text to encode
            allowed_special (set|None): A set of special tokens to allow pass through. If None, special handling is disabled.
        Returns:
            (list[int]): List of token IDs
        """
        # --- This section is to mimic tiktoken in terms of allowed special tokens----
        specials_in_vocab=[
            tok for tok in self.inverse_vocab if tok.startswith("<|") and tok.endswith("|>")
        ]
        if allowed_special is None: # nothing is allowed
            disallowed=[tok for tok in specials_in_vocab if tok in text]
            if disallowed: raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")
        else: # some specific tokens are allowed (e.g., we use <|endoftext|>)
            disallowed=[tok for tok in specials_in_vocab if tok in text and tok not in allowed_special]
            if disallowed: raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")
        #-------------------------------------------------------------------------------
        
        token_ids=[]
        # if some specials are allowed, split around them and passthrough those ids
        if allowed_special is not None and len(allowed_special)>0:
            # e.g., allowed_special={"<|endoftext|>", "<|im_start|>", "<|im_end|>"} will give
            # special_pattern='(<\\|endoftext\\|>|<\\|im_start\\|>|<\\|im_end\\|>)'
            special_pattern="("+"|".join(
                re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)
            )+")" 
        
            # iterate through each match, e.g., 
            # [<re.Match object; span=(42, 55), match='<|endoftext|>'>,<re.Match object; span=(86, 99), match='<|endoftext|>'>]
            last_index=0
            for match in re.finditer(special_pattern, text): 
                prefix=text[last_index:match.start()]
                token_ids.extend(self.encode(prefix, allowed_special=None)) # encode prefix normally
        
                special_token=match.group(0)
                if special_token in self.inverse_vocab: token_ids.append(self.inverse_vocab[special_token])
                else: raise ValueError(f"Special token {special_token} not found in vocabulary")
                last_index=match.end()
        
            text=text[last_index:] # process the remainder
        
            # extra guard for any other special literals left over
            disallowed=[tok for tok in self.inverse_vocab
                       if tok.startswith("<|") and tok.endswith("|>") and tok in text and tok not in allowed_special]
            if disallowed: raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")
        
        # --------newline and carriage return handling----------------------------------
        tokens=self.pretokenize_text(text, replace_space=self.replace_space)
        #-------------------------------------------------------------------------------
        # map tokens to IDs (BPE if needed)
        for tok in tokens:
            if tok in self.inverse_vocab: token_ids.append(self.inverse_vocab[tok])
            else: token_ids.extend(self.tokenize_with_bpe(tok))
        
        return token_ids

    def decode(self, token_ids):
        """Decode a list of token IDs back into a string
        Args:
            token_ids (list[int]): List of token IDs to decode
        Returns:
            (str): Decoded string
        """
        out=[]
        for tid in token_ids:
            if tid not in self.vocab: raise ValueError(f"Token ID {tid} not found in vocab")
            tok=self.vocab[tid]
        
            # map GPT-2 special characters back to real characters
            if (self.bpe_ranks and tid==198) or tok=="\n": out.append('\n')
            elif (self.bpe_ranks and tid==201) or tok=='\r': out.append('\r')
            elif self.replace_space and tok.startswith(self.replace_space): out.append(" "+tok[1:])
            else: out.append(tok)
        return "".join(out)
    
    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """Save the vocabulary and BPE meregs to JSON files
        Args:
            vocab_path (str): Path to save the vocabulary
            bpe_merges_path (str): Path to save the BPE merges
        """
        # save vocabulary
        with open(vocab_path, 'w', encoding='utf-8') as file: json.dump(self.vocab, file, ensure_ascii=False, indent=2)
        
        # save BPE merges as a list of dictionaries
        with open(bpe_merges_path, 'w', encoding='utf-8') as file:
            merges_list=[{'pair':list(pair), "new_id":new_id} for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """Load the vocabulary and BPE merges from JSON files
        Args:
            vocab_path (str): Path to the vocabulary file
            bpe_merges_path (str): Path to the BPE merges file
        """
        # load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as file:
            loaded_vocab=json.load(file)
            self.vocab={int(k):v for k, v in loaded_vocab.items()} # both key and values were read as str
            self.inverse_vocab={v:int(k) for k, v in loaded_vocab.items()}
        
        # load BPE merges
        with open(bpe_merges_path, 'r', encoding='utf-8') as file:
            merges_list=json.load(file)
            for merge in merges_list:
                pair=tuple(merge['pair']) # merge['pair'] were read as list[int,int]
                new_id=merge['new_id'] # merge['new_id'] was read as int
                self.bpe_merges[pair]=new_id