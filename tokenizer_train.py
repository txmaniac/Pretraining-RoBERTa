from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import os
import sys

if __name__ == "__main__":
    list_of_args = sys.argv[1:]

    paths = [list_of_args[0]]

    tokenizer = ByteLevelBPETokenizer(lowercase=True)

    paths = []
    # Customize training
    tokenizer.train(files=paths, vocab_size=30000, min_frequency=2,
                    show_progress=True,
                    special_tokens=[
                                    "<s>",
                                    "<pad>",
                                    "</s>",
                                    "<unk>",
                                    "<mask>",
    ])
    #Save the Tokenizer to disk
    tokenizer.save_model(list_of_args[1])