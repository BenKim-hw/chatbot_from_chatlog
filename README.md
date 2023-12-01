# Chatbot_from_chatlog
Create a chatbot from your Kakaotalk chatlog using [KoGPT2](https://github.com/SKT-AI/KoGPT2)
The default model is [KoGPT2](https://github.com/SKT-AI/KoGPT2), a pretrained Korean version of [GPT-2](https://openai.com/research/better-language-models). For chatlogs in other languages, please select appropriate models for that language.
Some parts of the code are imported from a [free wikidocs book on NLP](https://github.com/ukairia777/tensorflow-nlp-tutorial)

# How to use
## Directory tree
```bash
├── chatlogs_raw
│   ├── chatlog1.csv
├── chatlogs_processed
│   ├── chatlog1_processed.csv
├── models
│   ├── chatlog1_loss.pt
├── utils
│   ├── __init__.py
│   └── parse.py
│   └── preprocessing.py
├── main.py
└── chatbot.py

```

## Train
```
python main.py --train -tf [Name of chatlog] -te [Train epochs]
```

- -t, --train = training mode
- -tf, --train_file = Name of training chatlog (minus the .csv)
- -te, --train_epochs = Number of training epochs *Default value* = 15

## Run
```
python main.py --run -rm [Name of model]
```

- -r, --run = runing mode
- -rm, --run_model = Name of model (minus the .pt)

# License
The models were relased under the following licenses:
- KoGPT2: CC BY-NC-SA 4.0 DEED License
