# Check if any relation exists between a given cause and effect variable pair (using Masked Language Models)

Given a cause and effect variable, this code finds if 
there is any relation at all between the cause and effect variable. Relations 
can be of the type : promotes, inhibits, or causes relation. (refer `data/verbs.py` for a full list)
 

For example for the sentence `education improves income`, this application uses several 
[masked language models](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ) 
to check if there is a positive probability of the token `income` to occur at the end of the sentence:
`education improves ______` etc.
If the probability is more than a user specified threshold value, the application will print that there is a relation.

 Note: instead of checking for any relation, if you would rather get into details and get all average probabilities pass `--use_polarity True`
 

 

   
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
When you use the command line argument `--use_polarity True` this code will become the same as
 `calc_relation_probabilities_from_commandline_args` . Refer README_commandline_input.md for details.
 

``` 
python check_relation_exists.py --cause education_level --effect income --models bert-base-cased --threshold 0.001 --use_polarity True
```

### Details of Inputs


Consider you want to model the probability of yield to occur at the end of the sentence: `weather improves yield` .

Then these are the expected inputs for this application:

- causal variables (e.g.,weather)
- effect variables (e.g.,yield)
- trigger verbs (e.g.,improves)
- masked language models (list of [Masked Language Models](https://keras.io/examples/nlp/masked_language_modeling/)) you want your prediction probabilities to be averaged across.)


##### Causal and effect variables:
 
 - should be provided after the command line arguments of `--cause` and `--effect` respectively.
 
 e.g.,`--cause education`
 
 - if a causal/effect variable is multi token, use underscore `_` to separate them e.g.,`education_level`



##### Trigger verbs
the verbs you want cause and effect variables to be measured by `--triggers` i.e the
machine will take average of probability of y to occur in sentences like  `x improves y`, `x accelerates y` etc. where improves accelerates etc are the triggers.
If no input is provided for `--triggers` by default all the verbs given in `data/verbs.py` are used as trigger verbs. 
### Masked language models

- list of masked language models that you want your prediction be averaged across.
- This should be provided after the command line argument of `--models` and must be separated by space 
e.g.,`--models bert-base-cased distilbert-base-uncased`
- Even though we have tested with `--models bert-large-uncased bert-large-cased bert-base-uncased bert-base-cased`  you should ideally be able to use most of the models given in the [list of models](https://huggingface.co/models) by huggingface co. 
- parameter `--models` is optional. If no model is provided by default `bert-base-cased` will be used.



### Outputs

- The code will print if any relation is found after checking all the probabilities.
 

    
e.g.,: 

`
No relation exists between ['weather'] to ['yield'] for the given threshold;
`

`found that there is a relation.  weather improves  yield with probability of 0.0031075812876224518 which is > given threshold:0.001
`


- or if you used `--use_polarity True`
```
average probabilities from causes to effect=3.653426321963909e-07
average probabilities from effects to causes=1.250469520325876e-05
```


### Contact:
Please contact mithunpaul@email.arizona.edu or msurdeanu@email.arizona.edu for any questions.
