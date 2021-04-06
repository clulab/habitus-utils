from calc_relation_probabilities_from_disk_inputs import calc_average_probabilities_across_synonyms
import argparse
from transformers import AutoModelForMaskedLM, AutoTokenizer

def rep_underscore_space(input_list):
    for index,each_element in enumerate(input_list):
        input_list[index]=each_element.replace("_"," ")

def parse_arguments():
    argparser = argparse.ArgumentParser("to parse causal documents")
    argparser.add_argument("--causes", nargs='+', help="list of cause variables and their synonyms as strings e.g., education education_standard",)
    argparser.add_argument("--effects",nargs='+', help="list of effect variables and their synonyms as strings e.g., ['income', 'income level']")
    argparser.add_argument("--triggers",nargs='+', help="list of trigger_verbs as strings e.g., ['improves', 'accelerates', 'boosts']")
    argparser.add_argument("--models",nargs='+', help="list of names of masked language models as strings e.g., ['bert-base-cased','distilbert-base-uncased']")
    args = argparser.parse_args()
    rep_underscore_space(args.causes)
    rep_underscore_space(args.effects)
    rep_underscore_space(args.triggers)
    rep_underscore_space(args.models)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    MLM_MODELS=args.models
    for each_model in MLM_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(each_model)
        model = AutoModelForMaskedLM.from_pretrained(each_model)
        avg_prob=calc_average_probabilities_across_synonyms(args.causes, args.effects, args.triggers, model, tokenizer)
        print(f"avg_prob={avg_prob}")
    #print(calc_average_probabilities_across_models(overall_prob_averages_across_models, output_file_overall))

