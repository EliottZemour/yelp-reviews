## Fine tuning sequence-to-sequence models (BART, T5)

This repository contains a script taken from [huggingface/transformers/tree/main/examples/pytorch/summarization](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization)

## Data
You can find the yelp reviews dataset as json file [here](https://www.yelp.com/dataset). The `yelp_academic_dataset_review.json` file is approximately 5Gb.

### Setup

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
> [!NOTE]
> A GPU is recommended for training. If you don't have one, you can use a free GPU from [Google Colab](https://colab.research.google.com/).

### Fine tuning

The goal is to generate synthetic columns containing natural language with seq2seq models. Lets take as an example the following data (from Yelp), with the `text` column being the one we want to generate (conditionnaly on other columns values):

| review id | stars | useful | funny | cool | text |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 5 | 1 | 0 | 1 | "Wow! Yummy, different, delicious. Our favorite is the lamb curry and korma. With 10 different kinds of naan!!!  Don't let the outside deter you (because we almost changed our minds)...go in and try something new! You'll be glad you did!"

For training, what you need is a csv file with two columns: `input` and `target`. The `input` column contains the text to condition the generation on, and the `target` column contains the text to generate. In our case, the `input` column will contain the values of the other columns, and the `target` column will contain the `text` column.

|                     input                    	|                               target                              	|
|:--------------------------------------------:	|:-----------------------------------------------------------------:	|
| "Generate review: stars: 5, useful: 1, funny: 0, cool: 1" 	| "Wow! Yummy, different, delicious. Our favorite is the lamb curry and korma. [...]" 	|

Once you have your files `train.csv` and `valid.csv` containing these two `str` columns, symply launch the training script with the following command:

```
python train_seq2seq.py config.json
```

the config.json file contains the following:
``` json
{
    "model_name_or_path": "facebook/bart",
    "train_file": "train.csv",
    "valid_file": "valid.csv",
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "do_train": true,
    "do_eval": true,
    "push_to_hub": false,
    "output_dir": "bart-finetuning",
    "overwrite_output_dir": true,
    "num_train_epochs": 2,
    "evaluation_strategy": "steps",
    "eval_steps": 10,
    "save_strategy": "epoch",
    "warmup_steps": 200,
    "gradient_checkpointing": true,
    "learning_rate": 1e-5,
    "fp16": false
}
```

Feel free to modify it as you like. At the end of each epoch, a model checkpoint should be saved in the directory specified by `output_dir`.

### Inference


``` python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = 'path_to_your_checkpoint'

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


text = "Generate review: stars: 5, useful: 1, funny: 0, cool: 1"
inputs = tokenizer(text, return_tensors='pt')

# you can play with the parameters and the sampling/decoding strategy here
out = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    num_beams=10,
    # do_sample=True,
    # temperature = 1.2,
    # top_p=0.8,
    num_return_sequences=5,
    length_penalty=5
)

gen_texts = []
for gen in gen_texts:
    gen_texts.append(tokenizer.decode(gen, skip_special_tokens=True))

for q in gen_texts:
    print(q)
```
