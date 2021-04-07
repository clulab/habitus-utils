# Calculating Directionality of Cause and Effect Variables using Masked Language Models
This application uses various 
[masked language models](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ) 
to find the probability of each effect token to appear at the end of a sentence which has a causal token with verb.

For example for the sentence `education improves income`, this application will
find probability of the token `income to occur at the end of :
        education improves ______ etc.

Note that if either of the tokens are multiworded tokens (e.g.; income level) probabilities are recursively
averaged. For example, for the sentence ` education improves income level` overall probability of `income level` to
occur at the end are:average(prob(education improves income [MASK]), prob(education improves [MASK])

This is useful in modeling various causal and effect statements which can then be used for many other downstream tasks like
data modeling.

## Steps
 ```
    conda create -n directionality python=3
    conda activate directionality
    pip install -r requirements.txt   
    conda install pytorch torchvision torchaudio -c pytorch *
    mkdir outputs 
    
```
*please install [pytorch](http://pytorch.org/) based on your machine configuration.

###### To execute :

`python  calc_relation_probabilities.py [input_file]`

e.g.,:
 `python  calc_relation_probabilities.py data/inputs.tsv    `
### Inputs

There are the expected inputs for this application. 

- causal variables
- effect variables
- trigger verbs
- masked language model


##### Causal and effect variables:
 
 - Input variables are to be provided in the file `data/inputs.tsv`
 - Each group of synonyms of variables must be provided in a newline and should be separated by a tab

e.g.,
```
1	education	education standard
2	income	income level
```


##### Trigger verbs

- Trigger verbs should be provided in the file data/verbs.py as a comma separated list.

e.g.,
```
all_promote_verbs = ["improves", "accelerates", "boosts", "compounds", "enhances", "escalates",
                           "increases", "facilitates", "spikes"]
```

### Masked language models

- list of masked language models that you want to be averaged across should be mentioned in MLM_MODELS in `./calc_relation_probabilities.py`
- You can add the name of any models given in the [list of models](https://huggingface.co/models) by huggingface co. 

    e.g.,:
`MLM_MODELS=["bert-base-uncased","distilbert-base-uncased","bert-large-uncased","bert-base-cased"]`

Note: if no models are specified `distilbert-base-uncased` will be used by default.


### Outputs




- There are two types of output files
    - probabilities per model: `outputs/model_name_probabilities.tsv`
    - probabilities across models: `outputs/overall_probabilities.tsv`


- Each line in the output file will be the average probabilities across cartesian products of 
    - each synonym of input,
    - each line of input,
    - each trigger verb,
    - each model

    
e.g.,: 
`
1	2	PROMOTES	0.0003255476736507231
`

means the average probability of the sentence created by input variable 2 (all income related synonyms) to occur 
at the end of input variable 1 (all education related synonyms) across all PROMOTE VERBS




### Contact:
Please contact mithunpaul@email.arizona.edu for any questions.
