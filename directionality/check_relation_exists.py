import calc_relation_probabilities_from_commandline_args
from calc_relation_probabilities_from_commandline_args import DirectionalProbabilities,replace_underscore_with_space
import argparse
from data.verbs import *

def parse_arguments():
        argparser = argparse.ArgumentParser("to parse causal documents")
        argparser.add_argument("--cause", nargs='+',
                               help="cause variables  e.g., education ", )
        argparser.add_argument("--effect", nargs='+',
                               help="effect variable e.g., income ")
        argparser.add_argument("--models", nargs='+', default=['distilbert-base-uncased'],
                               help="list of names of masked language models as strings separated by space  e.g., bert-base-cased distilbert-base-uncased")
        argparser.add_argument("--use_polarity", type=bool, default=False,
                               help="the threshold value, below which probabilities are considered as no relation")
        argparser.add_argument("--threshold", type=float, default=0.0002,
                               help="the threshold value, below which probabilities are considered as no relation")
        args = argparser.parse_args()
        replace_underscore_with_space(args.cause)
        replace_underscore_with_space(args.effect)
        return args


def merge_all_verbs():
    all_verbs=[]
    for k,v in promote_inhibit_causal_triggers.items():
        all_verbs+=v
    return all_verbs

if __name__ == "__main__":
    '''
    --use-polarity = true -> old behavior (refer README_commandline_input.md)
    --use-polarity = false -> this one (refer README_check_relation_exists.md)

    '''
    args = parse_arguments()
    if(args.use_polarity==True):
        calc_relation_probabilities_from_commandline_args.main()
    else:
        all_verbs=merge_all_verbs()
        average_calculator=DirectionalProbabilities(causes=args.cause, effects=args.effect, models=args.models, triggers=set(all_verbs))
        dir1,dir2 = average_calculator.get_prob_across_models(threshold=args.threshold, check_directionality=True)
        if (dir1>args.threshold) or (dir2>args.threshold):
            print(f"found that some relation between {args.cause} to {args.effect} exists")
            exit()
