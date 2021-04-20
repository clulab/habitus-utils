from calc_relation_probabilities_from_disk_inputs import calc_average_probabilities_across_synonyms
import argparse
from transformers import AutoModelForMaskedLM, AutoTokenizer
from data.verbs import all_promote_verbs

class DirectionalProbabilities:
    def __init__(self,causes,effects, triggers,models):
        self.causes = causes
        self.effects = effects
        self.triggers = triggers
        self.models = models

    def calc_list_avg(self,list_averages):
        return float(sum(list_averages)) / float(len(list_averages))

    def get_prob_across_models(self):
        dir1_probs=[]
        dir2_probs = []
        for each_model in self.models:
            tokenizer = AutoTokenizer.from_pretrained(each_model)
            model = AutoModelForMaskedLM.from_pretrained(each_model)
            avg_prob = calc_average_probabilities_across_synonyms(self.causes, self.effects, self.triggers, model,tokenizer)
            dir1_probs.append(avg_prob)
            avg_prob_rev = calc_average_probabilities_across_synonyms(self.effects, self.causes, self.triggers, model, tokenizer)
            dir2_probs.append(avg_prob_rev)
        dir1_avg=self.calc_list_avg(dir1_probs)
        dir2_avg = self.calc_list_avg(dir2_probs)
        return (dir1_avg,dir2_avg)

def replace_underscore_with_space(input_list):
        for index,each_element in enumerate(input_list):
            input_list[index]=each_element.replace("_"," ")

def parse_arguments():
        argparser = argparse.ArgumentParser("to parse causal documents")
        argparser.add_argument("--causes", nargs='+', help="list of cause variables and their synonyms as strings separated by space e.g., education education_standard",)
        argparser.add_argument("--effects",nargs='+', help="list of effect variables and their synonyms as strings separated by space e.g., income income_level")
        argparser.add_argument("--triggers",nargs='+', help="list of trigger_verbs as strings separated by space e.g., improves accelerates boosts")
        argparser.add_argument("--models",nargs='+',default=['distilbert-base-uncased'], help="list of names of masked language models as strings separated by space  e.g., bert-base-cased distilbert-base-uncased")
        args = argparser.parse_args()
        if (args.triggers):
            pass
        else:
            args.triggers=all_promote_verbs
        replace_underscore_with_space(args.causes)
        replace_underscore_with_space(args.effects)
        return args

if __name__ == "__main__":
    args = parse_arguments()
    average_calculator=DirectionalProbabilities(args.causes, args.effects, args.triggers, args.models)
    dir1,dir2 = average_calculator.get_prob_across_models()
    print(f"average probabilities from causes to effect={dir1}")
    print(f"average probabilities from effects to causes={dir2}")


