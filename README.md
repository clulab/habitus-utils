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

## Pre requisites
 ```
    conda create -n directionality python=3
    conda activate directionality
    pip install -r requirements.txt   
    conda install pytorch torchvision torchaudio -c pytorch *
    mkdir outputs 
    
```
*please install [pytorch](http://pytorch.org/) based on your machine configuration.

## To execute this application

`python  python calc_relation_probabilities.py [input_file]`

For example:
 `python  python calc_relation_probabilities.py data/inputs.tsv    `
## Inputs

There are the expected inputs for this application. 

- causal variables
- effect variables
- trigger verbs
- masked language model


#####Causal and effect variables:
 
 - Input variables are to be provided in the file `data/inputs.tsv`
 - Each group of synonyms of variables must be provided in a newline and should be separated by a tab

e.g.,
```
1	education	education standard
2	income	income level
```
- The application will then calculate an average of probabilities across each line

e.g.,
probability of education promotes income = 
average of probabilities of ()education improves income, education standard improves income)

- and also across cartesian product of cause effect variable sentences
   e.g.,    
   ```
   education standard improves income   
   education standard improves income level
   ```


#####Trigger verbs

- Trigger verbs should be provided in the file data/verbs.py

Note: In this version (v1.0) even though we provide verbs which are not
`promotes` or `inhibits` like, (e.g.,all_does_not_cauase_verbs) they are 
not used in probability calculations.

By default distilbert is used as the masked language model. You can change this by changing the value of MLM_MODELS in `./calc_relation_probabilities.py`

You can add the name of any models given in the [list of models](https://huggingface.co/models)produ. 

e.g.,:
`MLM_MODELS=["bert-base-uncased","distilbert-base-uncased","bert-large-uncased","bert-base-cased"]`



## Outputs
######todo: update based on final data format you decide on

The outputs are the probabilities of each combination of the aforementioned
variables to occur with the list of provided verbs

The outputs will be found in `outputs/prob.json`

### Logging

Log files which can be used for debugging purposes and sentence level
 probability details can be found in `logs/*.log.`

Names of log files will include the input variables, type of masked language model used,
and the date of execution of this program.

e.g.,`numberofyearsoffarming_income_distilbert-base-uncasedMar-15-2021.log`

Logging levels (e.g., DEBUG, INFO, WARNING, ERROR etc.) can be set using the variable LOGGING_LEVEL in `./direction_validation.py`


