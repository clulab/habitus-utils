import calc_relation_probabilities_from_commandline_args
from calc_relation_probabilities_from_commandline_args import DirectionalProbabilities,replace_underscore_with_space,calc_prob
import argparse
from argparse import *
from data.verbs import *


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
        argparser = argparse.ArgumentParser("to parse causal documents")
        argparser.add_argument("--cause", nargs='+',
                               help="cause variables  e.g., education ", )
        argparser.add_argument("--effect", nargs='+',
                               help="effect variable e.g., income ")
        argparser.add_argument("--models", nargs='+', default=['bert-base-cased'],
                               help="list of names of masked language models as strings separated by space  e.g., bert-base-cased distilbert-base-uncased")

        argparser.add_argument("--threshold", type=float, default=0.0002,
                               help="the threshold value, below which probabilities are considered as no relation")

        '''--use-polarity = true -> old behavior (refer README_commandline_input.md)
            --use-polarity = false -> this one (refer README_check_relation_exists.md)
        '''
        argparser.add_argument("--use_polarity", type=str2bool, default=False,
                               help="do you want to do more than check if a relation exists-i.e get details of probabilities ? if yes, pass --use_polarity")
        args = argparser.parse_args()
        replace_underscore_with_space(args.cause)
        replace_underscore_with_space(args.effect)
        return args



if __name__ == "__main__":

    args = parse_arguments()
    if(args.use_polarity==True):
        calc_prob(args.cause, args.effect, set(all_promote_verbs), args.models)
    else:
        #all_verbs=merge_all_verbs()
        average_calculator=DirectionalProbabilities(causes=args.cause, effects=args.effect, models=args.models, triggers=set(all_verbs))
        dir1,dir2 = average_calculator.get_prob_across_models(threshold=args.threshold, check_directionality=True)
        if (dir1>args.threshold) or (dir2>args.threshold):
            print(f"found that some relation between {args.cause} to {args.effect} exists")
            exit()
        else:
            print(f"No relation exists between {args.cause} to {args.effect} for the given threshold")
