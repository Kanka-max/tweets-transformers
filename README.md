# tweets-transformers
Transformer For Tweets

This repository consists of notebooks that implement **zero-shot** classification using language transformers. The notebooks also implement data stories for various model performances with their visualizations rendered by **Plotly**.

The models documentation can be found here:
- [RoBERTa-Large-XNLI](https://huggingface.co/joeddav/xlm-roberta-large-xnli)
- [Facebook/BART-Large](https://huggingface.co/facebook/bart-large)

There's also python scripts to be deployed as a data app on [Stream Cloud](https://streamlit.io/cloud)


###### Getting started:

- Installation
```
!pip install transformers==3.1.0
```
- Model evaluation dependencies
```
!pip install evaluate
```

- Initialize roberta-large model
```
classifier = pipeline("zero-shot-classification", 
                      model='joeddav/xlm-roberta-large-xnli')
```

- Initialize facebook model

```
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
```