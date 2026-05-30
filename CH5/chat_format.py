from llm_from_scratch.CH5.utils import Tokenizer

# Llama3 should ideally be used with the correct prompt template that was used during finetuning 
class ChatFormat:

    def __init__(self, tokenizer:Tokenizer, *, default_system="You are a helpful assistant."):
        # Any argument passed after `*` must be a keyword argument, i.e., cannot be passed as a positional argument
        self.tok=tokenizer
        self.default_system=default_system

    def _header(self, role):
        """Encode <|start_header_id|>role<|end_header_id|>\n\n"""
        return ( [self.tok.special["<|start_header_id|>"]] + self.tok.encode(role) + 
                 [self.tok.special["<|end_header_id|>"]] + self.tok.encode("\n\n") )

    def encode(self, user_message, system_message=None, allowed_special=None):
        sys_msg=system_message if system_message is not None else self.default_system

        ids=[self.tok.special["<|begin_of_text|>"]]

        # system
        ids+=self._header("system")
        ids+=self.tok.encode(sys_msg)
        ids+=[self.tok.special["<|eot_id|>"]]

        # user
        ids+=self._header("user")
        ids+=self.tok.encode(user_message)
        ids+=[self.tok.special["<|eot_id|>"]]

        # assistant header (no content yet)
        ids+=self._header("assistant")

        return ids


def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
    # Find the index of the first occurence of <|end_header_id|>
    index=text.find(header_end)

    # return substring starting after <|end_header_id|>
    if index!=-1: return text[index + len(header_end):].strip() # remove leading/trailing whitespace
    else: return text # if the token is not found, return the original text
    