from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
import sys

if __name__ == "__main__":
    list_of_args = sys.argv[1:]
    #   argv[1]: path where model should be downloaded
    #   argv[2]: path where tokenizer should be downloaded

    model_path = list_of_args[0]
    tokenizer_path = list_of_args[1] if len(list_of_args) > 1 else model_path

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    config = RobertaConfig.from_pretrained("roberta-base")

    model = RobertaForMaskedLM.from_pretrained("roberta-base", config=config)

    tokenizer.save_pretrained(tokenizer_path)
    model.save_pretrained(model_path)