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

```
python calc_relation_probabilities_from_commandline_args.py --causes education education_standard --effects income income_level --triggers improves accelerates --models bert-base-cased distilbert-base-uncased
```


### Details of Inputs


Consider you want to model the probability of yield to occur at the end of the sentence: `weather improves yield` .

Then these are the expected inputs for this application:

- causal variables (e.g.,weather)
- effect variables (e.g.,yield)
- trigger verbs (e.g.,improves)
- masked language models (list of [Masked Language Models](https://keras.io/examples/nlp/masked_language_modeling/)) you want your prediction probabilities to be averaged across.)


##### Causal and effect variables:
 
 - should be provided after the command line arguments of `--causes` and `--effects` respectively.
 
 e.g.,`--causes education education_standard`
 - each causal/effect variable must be separated from another using a space e.g., `education weather`
 
 - if a causal/effect variable is multi token, use underscore `_` to separate them e.g.,`education_level`



##### Trigger verbs

- Trigger verbs should be provided after the command line argument `--triggers`.

e.g.,
```
--triggers improves accelerates 

```

Note: trigger verbs are optional. If no trigger verbs are provided, code will default to the `all_promote_verbs`  in 
`data/verbs.py`

### Masked language models

- list of masked language models that you want your prediction be averaged across.
- This should be provided after the command line argument of `--models` and must be separated by space 
e.g.,`--models bert-base-cased distilbert-base-uncased`
- You can add the name of any models given in the [list of models](https://huggingface.co/models) by huggingface co. 
- Masked models are optional. If no model is provided by default `distilbert-base-uncased` will be used.



### Outputs

- The code will output the probability in both directions. For example for the causal and effect variable example given above (weather promotes yield), the output will be 
    - probability of yield to occur at the end of weather promotes ________.
    - probability of weather to occur at the end of yield promotes ________.    
- Output for command line based code will be printed in command line itself.


    
e.g.,: 
```
average probabilities from causes to effect=0.0006510953470650647

average probabilities from effects to causes=0.002261902516086896
```





### Contact:
Please contact mithunpaul@email.arizona.edu or msurdeanu@email.arizona.edu for any questions.
