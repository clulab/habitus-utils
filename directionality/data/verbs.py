import itertools

all_promote_verbs = [
    "improves",
    "accelerates",
    "boosts",
    "compounds",
    "enhances",
    "escalates",
    "increases",
    "facilitates",
    "spikes",
]
all_inhibits_verbs = [
    "diminishes",
    "drops",
    "drains",
    "exhausts",
    "impairs",
    "inhibits",
    "hampers",
    "hinders",
    "eliminates",
    "disrupts",
]
all_causal_verbs = [
    "influences",
    "affects",
    "alters",
    "modifies",
    "impacts",
    "changes",
    "displaces",
]

# in negation you want DOES NOT INFLUENCE instead of DOES NOT INFLUENCES
all_does_not_promote_verbs = [
    "does not improve",
    "does not accelerate",
    "does not boost",
    "does not compound",
    "does not enhance",
    "does not escalate",
    "does not increase",
    "does not facilitate",
    "does not spike",
]
all_does_not_inhibits_verbs = [
    "does not diminish",
    "does not drop",
    "does not drain",
    "does not exhaust",
    "does not impair",
    "does not inhibit",
    "does not hamper",
    "does not hinder",
    "does not eliminate",
    "does not disrupt",
]
all_does_not_cauase_verbs = [
    "does not influence",
    "does not affect",
    "does not alter",
    "does not modify",
    "does not impact",
    "does not change",
    "does not displace",
]

all = [
    all_promote_verbs,
    all_inhibits_verbs,
    all_causal_verbs,
    all_does_not_promote_verbs,
    all_does_not_inhibits_verbs,
    all_does_not_cauase_verbs,
]
all_verbs = list(itertools.chain.from_iterable(all))

promote_inhibit_triggers = {
    "PROMOTES": all_promote_verbs,
    "INHIBITS": all_inhibits_verbs,
    "DOES_NOT_PROMOTE": all_does_not_promote_verbs,
    "DOES_NOT_INHIBT": all_does_not_inhibits_verbs,
}

promote_inhibit_causal_triggers = {
    "PROMOTES": all_promote_verbs,
    "INHIBITS": all_inhibits_verbs,
    "CAUSAL": all_causal_verbs,
    "DOES_NOT_PROMOTE": all_does_not_promote_verbs,
    "DOES_NOT_INHIBT": all_does_not_inhibits_verbs,
    "DOES_NOT_CAUSE": all_does_not_cauase_verbs,
}
