from calc_relation_probabilities_from_commandline_args import DirectionalProbabilities,replace_underscore_with_space
from calc_relation_probabilities_from_disk_inputs import names_triggers
import argparse

def parse_arguments():
        argparser = argparse.ArgumentParser("to parse causal documents")
        argparser.add_argument("--cause",
                               help="cause variables  e.g., education ", )
        argparser.add_argument("--effect",
                               help="effect variable e.g., income ")
        argparser.add_argument("--models", nargs='+', default=['distilbert-base-uncased'],
                               help="list of names of masked language models as strings separated by space  e.g., bert-base-cased distilbert-base-uncased")
        argparser.add_argument("--threshold", type=int, default=0.0002,
                               help="the threshold value, below which probabilities are considered as no relation")
        args = argparser.parse_args()
        args.cause.replace("_", " ")
        args.effect.replace("_", " ")
        return args


"""
      

       :param cause: cause variable (e.g., education)
       :param effect: effect variable (e.g.. income)
       :param triggers: all triggers (promotes, inhibits, causes)
       :param threshold : the threshold value, below which probabilities are considered as no relation
       :return: boolean, True or False, has relation
       
"""

if __name__ == "__main__":
    args = parse_arguments()
    for k,v in names_triggers.items():
        average_calculator=DirectionalProbabilities(causes=args.cause, effects=args.effect, models=args.models, triggers=set(v))
        dir1,dir2 = average_calculator.get_prob_across_models()
        if (dir1>args.threshold) or (dir2>args.threshold):
            print(f"found that some relation between {args.cause} to {args.effect} exists")
            exit()
