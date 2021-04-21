# Check if any relation exists between a given Cause and Effect Variables using Masked Language Models

Given a cause and effect variable, find if it has any relation at all. i.e doesnt have any promotes, inhibits, or causes relation.
 
 
This application uses various 
[masked language models](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ) 
to find the probability of each effect token to appear at the end of a sentence which has a causal token with verb. For example for the sentence `education improves income`, this application will
find probability of the token `income to occur at the end of :
        education improves ______ etc.

    
   note: threshold value. if the probability is below that consider it as no relation.
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
python check_relation_exists.py --cause education  --effect income   --models distilbert-base-uncased --threshold 0.001
```
Note: instead of checking for any relation, you would rather get into details and get average probabilities pass `--use_polarity`


```
python check_relation_exists.py --cause education  --effect income   --models distilbert-base-uncased --threshold 0.001 --use_polarity
```

### Details of Inputs


Consider you want to model the probability of yield to occur at the end of the sentence: `weather improves yield` .

Then these are the expected inputs for this application:

- causal variables (e.g.,weather)
- effect variables (e.g.,yield)
- trigger verbs (e.g.,improves)
- masked language models (list of [Masked Language Models](https://keras.io/examples/nlp/masked_language_modeling/)) you want your prediction probabilities to be averaged across.)


##### Causal and effect variables:
 
 - should be provided after the command line arguments of `--causes` and `--effect` respectively.
 
 e.g.,`--cause education`
 
 - if a causal/effect variable is multi token, use underscore `_` to separate them e.g.,`education_level`



##### Trigger verbs
given in `data/verbs.py`

### Masked language models

- list of masked language models that you want your prediction be averaged across.
- This should be provided after the command line argument of `--models` and must be separated by space 
e.g.,`--models bert-base-cased distilbert-base-uncased`
- You can add the name of any models given in the [list of models](https://huggingface.co/models) by huggingface co. 
- Masked models are optional. If no model is provided by default `distilbert-base-uncased` will be used.



### Outputs

- The code will print if any relation is found after checking all the probabilities.
 

    
e.g.,: 
```

```





### Contact:
Please contact mithunpaul@email.arizona.edu or msurdeanu@email.arizona.edu for any questions.
