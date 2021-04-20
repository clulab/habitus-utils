from calc_relation_probabilities_from_commandline_args import DirectionalProbabilities
from calc_relation_probabilities_from_disk_inputs import promote_inhibit_causal_triggers
import argparse

def parse_arguments():
        argparser = argparse.ArgumentParser("to parse causal documents")
        argparser.add_argument("--cause",
                               help="cause variables  e.g., education ", )
        argparser.add_argument("--effect",
                               help="effect variable e.g., income ")
        argparser.add_argument("--models", nargs='+', default=['distilbert-base-uncased'],
                               help="list of names of masked language models as strings separated by space  e.g., bert-base-cased distilbert-base-uncased")
        argparser.add_argument("--threshold", type=float, default=0.0002,
                               help="the threshold value, below which probabilities are considered as no relation")
        args = argparser.parse_args()
        args.cause.replace("_", " ")
        args.effect.replace("_", " ")
        return args

if __name__ == "__main__":
    args = parse_arguments()
    for k,v in promote_inhibit_causal_triggers.items():
        average_calculator=DirectionalProbabilities(causes=args.cause, effects=args.effect, models=args.models, triggers=set(v))
        dir1,dir2 = average_calculator.get_prob_across_models()
        print(f"*************")
        print(f"triggers:{set(v)}")
        print(f"average probabilities from cause to effect={dir1}")
        print(f"average probabilities from effect to cause={dir2}")
        if (dir1>args.threshold) or (dir2>args.threshold):
            print(f"found that some relation between {args.cause} to {args.effect} exists")
            exit()
