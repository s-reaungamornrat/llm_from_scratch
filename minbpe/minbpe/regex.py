"""
Minimal (byte-level) Byte-Pair Encoding tokenizer

Algorithmically follows along the GPT tokenizer
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from .base import Tokenizer, get_stats, merge, clean_text

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
# The optimized, secure modern standard
OPT_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}\p{M}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}\p{M}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT5_SPLIT_PATTERN= "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )

class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str->int dictionary of special tokens, e.g., {'<|endoftext|>':100257}
        """
        super().__init__()
        self.pattern=GPT5_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern=re.compile(self.pattern)
        self.special_tokens={}
        self.inverse_special_tokens={}

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str->int
        # example: {"<|endoftext|>":100257}
        self.special_tokens=special_tokens
        self.inverse_special_tokens={v:k for k, v in special_tokens.items()}

    def train(self, text, vocab_size, verbose=False):
        
        assert vocab_size>=256
        num_merges=vocab_size-256
        
        # clean text 
        print("In regex.train starts cleanning text")
        text=clean_text(text)
        print("In regex.train finishes cleanning text")
        
        # split the text up into text chunks, i.e., list of each individual words, marks, symbols, e.g., ['copy', 'waste', ',', ' as',...]
        text_chunks=re.findall(self.compiled_pattern, text) 
        
        # input text preprocessing, producing list of list[int], where each list[int] corresponding to each chunk (individual words, marks, ...)
        #ids=[list(ch.encode('utf-8')) for ch in text_chunks] 
        # this is slower than the above statement, but we want to see whether what causes errors if there are
        ids=[]
        for chunk in text_chunks: 
            try: chunk=chunk.encode('utf-8')
            except UnicodeEncodeError as err: print(f"UTF-8 encoding error for {chunk}\n{err}"); continue
            ids+=[ list(chunk) ]
        # iterative merge the most common pairs to create new tokens
        merges={} # (int, int)->
        vocab={idx:bytes([idx]) for idx in range(256)} # idx->bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats={}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            if not stats: break # no more frequent pairs
            # find the pair with the highest count
            pair=max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx=256+i
            # replace all occurrences of pair in ids with idx
            ids=[merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair]=idx
            vocab[idx]=vocab[pair[0]]+vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair}->{idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        
        # save class variables
        self.merges=merges # used in encode()
        self.vocab=vocab # used in decode()
        self.vocab_size=max(self.vocab.keys())+1
        print(f"In regex.train : {max(vocab.keys())=}, {min(vocab.keys())=}")

    def _encode_chunk(self, text_bytes):
        """
        Args:
            text_bytes (bytes): String of a word or subword that was converted to bytes, e.g., b'copy'
        Returns:
            (list[int]): Token IDs for the input after merging
        """
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0...255
        ids=list(text_bytes) # list[int]
        while len(ids)>2:
            # find the pair with the lowest merge index (i.e., most co-occurrent pair)
            stats=get_stats(ids)
            pair=min(stats, key=lambda p:self.merges.get(p, float('inf')))
            # subtle: if there are no more merges available, the key will result in an inf for every single pair
            # and the min will be just the first pair in the list, arbitrarily we can detect this terminating case by
            # a membership check
            if pair not in self.merges: break # nothing else can be merged anymore
            # otherwise, let's merge the best pair (lowest merge index)
            idx=self.merges[pair]
            ids=merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens. It breaks text into each chunk using regex pattern (i.e.,individual words, symbols,
        marks, etc), convert `str` to `bytes` and `list[int]` which is a list of IDs for each chunk. Then it iteratively merges pairs of token
        IDs based on pairs and IDs in `self.merges`. Finally, it combines lists of token IDs from each chunk into a single list of token IDs.
        Args:
            text (str): Consecutive strings forming text
        Returns:
            list[int]: 
        """
        # split text into chunks of text by categories defined in regex pattern
        text_chunks=re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids=[]
        for chunk in text_chunks:
            chunk_bytes=chunk.encode('utf-8') # raw bytes, e.g., 'copy' -> b'copy'
            chunk_ids=self._encode_chunk(chunk_bytes) # list[int], token ids for this chunk (word, mark, symbol)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """Unlike encode_ordinary, this function handles special tokens. 
        allowed_special: can be "all"|"none"|"none_raise" or a custome set of special tokens
        if none_raise, then an error is raised if any special tokens is encountered in text
        this is the default tiktoken behaviour right now as well. Any other behaviour is either annoying or a major footgun
        """
        # decode the user desire w.r.t handling of special tokens
        special=None
        if allowed_special=="all": special=self.special_tokens
        elif allowed_special=="none": special={}
        elif allowed_special=="none_raise": 
            special={}
            assert all(token not in text for token in re_tokenizer.special_tokens)
        elif isinstance(allowed_special, set):
            special={k:v for k, v in self.special_tokens.items() if k in allowed_special}
        else: raise ValueError(f"allowed_special={allowed_special} is not understood")
        
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        
        # otherwise, we have to be careful with potential special tokens in the text. We handle special tokens by spolitting the text
        # based on the occurrence of any exact match with any of the special tokens. We cab use re.split() for this. Note that surrounding 
        # the pattern with () makes it into a capturing group, so the special tokens will be included
        special_pattern="("+"|".join(re.escape(k) for k in special)+")" # e.g., '(<\\|endoftext\\|>|<\\|beginoftext\\|>)'
        # split special_tokens appeared in the text from the rest, e.g., text="<|beginoftext|>Hello world! You are great! <|endoftext|>"
        # ['', '<|beginoftext|>', 'Hello world! You are great! ', '<|endoftext|>', '']
        special_chunks=re.split(special_pattern, text) 
        # now all the special characters are separated from the rest of the text. All chunks of the text are encoded separately, then
        # results are joined
        ids=[]
        for part in special_chunks:
            if not part: continue
            if part in special:
                # this is a special token, encode it separatelt as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        """
        Args:
            ids (list[int]): List of token IDs
        Returns:
            (str): Decoded strings
        """
        # given ids (list of integers), return Python string
        part_bytes=[]
        for idx in ids:
            if idx in self.vocab: part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens: part_bytes.append(self.inverse_special_tokens[idx].encode('utf-8'))
            else: raise ValueError(f"Invalid token ID: {idx}")
        text_bytes=b"".join(part_bytes)
        text=text_bytes.decode("utf-8", errors='replace')
        return text