from functools import lru_cache
import regex

r"""
Under standard UTF-8 encoding, common English characters take up 1 byte, while others (emojis, math symbols, etc) can take up 3 to 4 bytes. There are 2 main problems
- Unknow token (UNK): If users use a rare emoji or character, the model cannot read it and replace it with an <UNK> token 
- Vocab bloat: If we try to include every possible Unicode character, the vocab size will skyrocket.
"""

@lru_cache
def byte_to_unicode():
    """Returns list of utf-8 bytes and a corresponding list of unicode strings.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab if you want to 
    avoid UNKs. When you are at something like a 10B token dataset you end up needing around 5K for decent coverage. This is a significant
    percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and unicode strings and avoid
    mapping to whitespace/control characters the bpe code barfs on

    In other words, this function creates a look-up table that maps all 256 raw bytes to 256 distinct, safe Unicode characters, 
    by leaving the standard readable characters (like a, b, 1, ?) alone), but for control characters, whitepaces and invisible bytes, 
    mapping them to obsecure, harmless Unicode characters (located in parts of the Unicode spectrum that does not mess with BPE). 
    For example,
    - Raw byte 32 (A literal space " ") maps to unicode character "Ġ"
    - Raw byte 10 (A literal newline "\n") maps to unicide character "Ċ"
    - Raw byte 97 (The letter a) stays Unicode character a
    """
    # note `ord` stands for ordinal, converting a single character string into its corresponding Unicode code point integer.
    # [33,127), [161,173), [174,256)
    # bytes_ will contain all possible 8-bit raw byte values. We initialize it with standard readable characters whose
    # mapping remain unchanged, e.g., A, !, a, 1 
    bytes_ = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    # Unicode string characters that bytes in `bytes` will safely map to.
    chars=bytes_[:] # shallow copy so changes to cs will not affect bs
    n=0
    for b in range(2**8): # from 0 to 255
        if b not in bytes_: # standard readable characters
            bytes_.append(b) 
            chars.append(2**8+n)
            n+=1
    # we note that `chr` converts an integer representing a Unicode code point into its corresponding string character,
    # i.e., inverse of `ord`
    chars=[chr(n) for n in chars]
    return dict(zip(bytes_, chars))

def get_pairs(word):
    """Return a set of symbol pairs in a word. Word is represented as tuple of symbols where
    symbols are variable-length strings
    Args:
        word (tuple[str]): Sequence of each character forming one token, e.g., 
            word=('T', 'r', 'a', 'n', 's', 't', 'h', 'y', 'r', 'e', 't', 'i', 'n') from the token 'Transthyretin'
    Returns:
        (set[tuple[str, str]]): Set of pairs of consecutive characters, e.g., {('T','r'), ('r','a'), ...}
    """
    pairs=set()
    prev_char=word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char=char
    return pairs
    
class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder=encoder # vocabulary {str:int}
        self.decoder={v:k for k, v in self.encoder.items()}
        self.errors=errors # how to handle errors in decoding
        # after convert text to raw UTF-8 bytes, the module calls `byte_to_unicode` to convert control bytes to readable unicode characters
        self.byte_encoder=byte_to_unicode()
        self.byte_decoder={v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks=dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache={}

        # should have added re.IGNORECASE so BPE merges can happen for capitialized versions of contractions
        # regular expression pattern for pre-tokenization. This regex slices text into distinct categories (words, contractions,
        # numbers, punctuation). The symbol `|` acts as an OR operator, separating the string into 7 matching groups.
        # - 's|'t|'re|'ve|'m|'ll|'d| separate "common English contractions" so they don't get fused to the base word
        # - ?\p{L}+ captures an optional space(?) followed by one or more Unicode letters (\p{L}+), typically capture the whole 
        #         word with leading space if exists, e.g., " Hello", "world", " Python"
        # - ?\p{N}+ captures an optional space(?) followed by one or more Unicode numeric digits (\p{N}+), grouping standalone numbers,
        #         e.g., " 42", "2026"
        # - ?[^\s\p{L}\p{N}]+ captures an optional space(?) followed by one or more characters that are not spaces, letters, or numbers, 
        #         isolating punctuation marks, symbols, emojis, and math signs anlong with an optional leading space, e.g., " !", ", ", "..."
        #         [] defines a character class. ^ is a negation operation, so 
        #         [^...] means "match any characters except the ones listed inside the brackets" 
        #             \s: any whitespace characters (spaces, tabs, newlines)
        #             \p{L}: any Unicode letters including A-Z and a-z and accebted characters (e.g., é, ü) Chinese, Arabic, etc.
        #             \p{N}: any Unicode number including 0-9, Roman numerals, fractions, and numeric scripts from other languages
        #         + matches one or more of the preceding element 
        # - \s+(?!\S) captures one or more whitespaces (\s+) only if they are not followed by a non-whitespace character 
        #         ((?!\S) is a negative lookahead), i.e., spaces at the very end of a line or document
        # - \s+ captures any remaining sequence of whitespace characters, catching structural text spacing (e.g., \n, \t, multiple consecutive spaces)
        self.pattern=regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", regex.IGNORECASE)
        # pad liekly stands for punctuation, alphabet, digits
    
    def bpe(self, token):
        """
        Args:
            token (str): Token of subword or word
        Returns:
            (str): Token broken into subword representable by vocabulary
        Example:
            >>> self.bpe('Transthyretin')
            'T ran st hy ret in'
        """
        
        if token in self.cache: return self.cache[token]
            
        word=tuple(token) # convert the word into tuple of characters
        pairs=get_pairs(word)
        
        if not pairs: return token
        
        while True:
            # get a pair with the lowest rank
            bigram=min(pairs, key=lambda pair:self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks: break
        
            first, second=bigram
            new_word=[]
            i=0
            while i<len(word):
                try:
                    # finds the index of the first occurrence of `first` in `word` starting the search from index `i`
                    j=word.index(first,i) # j is the index before `first`
                    # add all skipped elements up to the match
                    new_word.extend(word[i:j]) # from i to before the `first` match
                    i=j
                except ValueError: 
                    # `first` no loner exists in the remaining part of the word
                    new_word.extend(word[i:]); 
                    break
        
                # check whether the matching 'first' is followed by 'second'
                if word[i]==first and i<len(word)-1 and word[i+1]==second:
                    new_word.append(first+second) # add f"{first}{second}" instead of f"{first}", f"{second}"
                    i+=2 # merge successfully: skip past both `first` and `second`
                else: # if word[i+1]!=second (if `first` is not followed by `second`)
                    new_word.append(word[i]) # add `first`
                    i+=1 #  move past `first`
    
            word=tuple(new_word)
            if len(word)==1: break
            else: pairs=get_pairs(word)
        
        word=" ".join(word)
        self.cache[token]=word
        return word
            
    def encode(self, text):
        """
        Args:
            text (str): Input text to encode
        Returns:
            (list[int]): Sequence of token IDs
        Examples:
            >>> text=("Transthyretin Amyloid Cardiomyopathy (ATTR-CM):  A fatal, underdiagnosed condition caused "
                      "by abnormal protein buildup that stiffens the heart muscle, typically affecting older adult.")
            >>> self.encode(text)
            [51, 2596, 301, 12114, 1186, 259, 1703, 2645, 1868, 5172, 72, 9145, 27189, 357, 1404, 5446, 12, 24187, 2599, 220, 
            317, 10800, 11, 739, 47356, 1335, 4006, 4073, 416, 18801, 7532, 40502, 326, 15175, 641, 262, 2612, 8280, 11,
            6032, 13891, 4697, 4044, 13]
        """
        bpe_tokens=[]
        for token in regex.findall(self.pattern, text):
            token=''.join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self,tokens):
        """
        Args:
            tokens (list[int]): Sequence of token IDs
        Returns:
            (str): Decoded text
        """
        text="".join([self.decoder[token] for token in tokens])
        text=bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text