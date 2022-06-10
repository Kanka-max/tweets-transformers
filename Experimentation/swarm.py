import time 
from datetime import datetime
import streamlit as st
st.set_page_config(layout="wide",
                   page_title= "D-EQ Data App", page_icon=":sunny:" )
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from sklearn.metrics import hamming_loss
import transformers
from transformers import Trainer
import plotly.graph_objects as go
from transformers import pipeline
from tqdm import tqdm
import evaluation
from PIL import Image

with st.container():
    image = Image.open('sunset.jpg')

    st.image(image, caption='Bumblebee')

import streamlit.components.v1 as components

# load corpus and split them to hateful vs non-hateful
eng_tweets = "https://raw.githubusercontent.com/Kanka-max/tweets-transformers/main/Experimentation/en_200_examples.xlsx%20-%20en_200_examples.csv"
swa_tweets = "swa_tweets.csv"


eng_tweets = pd.read_csv(eng_tweets)
pipeline_one='joeddav/xlm-roberta-large-xnli'
pipeline_two = "facebook/bart-large-mnli"
def load_classifier(classifier):
    classifier = pipeline("zero-shot-classification", model= classifier)
    return classifier

#TITLE

st.image(image, caption='sunflower.jpg')
st.title("Language Transformers Data App :sunglasses:")
st.markdown("**Huggingface** models inferred on multilingual corpus for **zero-shot** text classification")

with st.container():
    option = st.selectbox(
     'Which dataset would you like to try?',
     ('English Corpus (Hateful)', 'English Corpus (Non-Hateful)', 
      'English Corpus (Hateful/Non-H)', 'Swahili Corpus (Hateful)', 'Swahili Corpus (Non-Hateful)', 
      'Swahili Corpus (Hateful/Non-H))', "Multilngual Multilabel",
      'Custom'))

with st.container():
    model = st.selectbox(
     'Which model would you like to train?',
     ('RoBERTa-Large', 
      'Facebook-BART-Large'))


###### SIDEBAR ####
# JUST DO IT
with st.container():

    st.sidebar.subheader("Train & Evaluate")
with st.container():
    sidebar_option = st.sidebar.selectbox(
     'Fit a model on a dataset?',
     ('English Corpus', 'Kiswahili', 'Cross-lingual','Sheng'))


if sidebar_option == "English Corpus" and model == "RoBERTa-Large":

    with st.spinner('Loading transformer classifier...'):
        classifier = load_classifier(pipeline_one)
        time.sleep(5)
        st.success('Model ready!')

    def train_model(df):

        category_map = {"hateful": "hate", "non-hateful": "loving"}

        candidate_labels = list(category_map.values())
 #candidate labels are basically the classes that the classifier will predict
        predictedCategories = []
        trueCategories = []
        for i in tqdm(range(200)):
            text = df.iloc[i,]['text']
            cat = df.iloc[i,]['label']
            cat = cat.split()
            res = classifier(text, candidate_labels)#setting multi-class as True
            labels = res['labels'] 
            scores = res['scores'] #extracting the scores associated with the labels
            res_dict = {label : score for label,score in zip(labels, scores)}
            sorted_dict = dict(sorted(res_dict.items(), key=lambda x:x[1],reverse = True)) #sorting the dictionary of labels in descending order based on their score
            categories  = []
            for i, (k,v) in enumerate(sorted_dict.items()):
                if(i > 2): #storing only the best 3 predictions
                    break
                else:
                    categories.append(k)
            predictedCategories.append(categories)
            trueCats = [category_map[x] for x in cat]
            trueCategories.append(trueCats)

            category_idx = {cat : i for i,cat in enumerate(category_map.values())} 
            y_trueEncoded = []
            y_predEncoded = []
            for y_true, y_pred in zip(trueCategories, predictedCategories):
                encTrue = [0] * len(category_map)
                for cat in y_true:
                    idx = category_idx[cat]
                    encTrue[idx] = 1
                y_trueEncoded.append(encTrue)
                encPred = [0] * len(category_map)
                for cat in y_pred:
                    idx = category_idx[cat]
                    encPred[idx] = 1
                    y_predEncoded.append(encPred)
        return y_trueEncoded, y_predEncoded
    if st.button('Evaluate'):


        with st.spinner('Evaluating...'):

            (y_trueEncoded, y_predEncoded) = train_model(eng_tweets)
            time.sleep(5)

            precision_metric = evaluate.load("precision")
            precision_r = precision_metric.compute(references=[0, 1,1,0], predictions=[1,1,1,1])

            matthews_metric = evaluate.load("matthews_correlation")
            matthews_r = matthews_metric.compute(references=[0,1,1,0], predictions=[1,1,1,1])

            f1_metric = evaluate.load("f1")
            f1_r = f1_metric.compute(references=[0,1,1,0], predictions=[1, 1, 1, 1])

            accuracy_metric = evaluate.load("f1")
            accuracy_r = accuracy_metric.compute(references=[0,1,1,0], predictions=[1, 1, 1, 1])
        
            hamming_r = hamming_loss(y_trueEncoded,y_predEncoded)
            st.success('Finished Evaluating Five Metrics!')


    def plot_evaluation(hamming_r, precision_r, accuracy_r, f1_r, matthews_r):
        fig = go.Figure()

        fig.add_trace(go.Bar(
                     x= hamming_r,
                     name = "hamming loss",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

        fig.add_trace(go.Bar(
                     x= precision_r,
                     name = "precision",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

        fig.add_trace(go.Bar(
                     x= accuracy_r,
                     name = "accuracy",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

        fig.add_trace(go.Bar(
                     x= f1_r,
                     name = "F1 Score",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

        fig.add_trace(go.Bar(
                     x= matthews_r,
                     name = "Matthews Correlation",
                     orientation = "h",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

        fig.update_layout(
            title_text = "Evaluation of xlm-roberta-large-xnli Language Model on English Hateful Tweets",
            title_font_size = 16,
            title_font_color = "crimson",
            title_font_family = "Gravitas One",
            xaxis=dict(
            title = "Month",
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        
        ),
            yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
            barmode='group',
            paper_bgcolor='rgb(248, 248, 255)',
            plot_bgcolor='rgb(248, 248, 255)',
            margin=dict(l=120, r=10, t=140, b=80),

            legend = dict(x=0, y=1.0, bgcolor = "LemonChiffon", bordercolor = "gold"))

        return st.plotly_chart(fig, sharing="streamlit", use_container_width=True)

    with st.container():
        plot_evaluation(hamming_r, precision_r, accuracy_r, f1_r, matthews_r)
        st.balloons()
if sidebar_option == "English Corpus" and model == "Facebook-BART-Large":

    classifier = load_classifier(pipeline_two)
    if st.button('Evaluate'):
        with st.spinner('Evaluating model..'):

            (y_trueEncoded, y_predEncoded) = train_model(eng_tweets)

            precision_metric = evaluate.load("precision")
            precision_results = precision_metric.compute(references=[0, 1,1,0], predictions=[1,1,1,1])

            matthews_metric = evaluate.load("matthews_correlation")
            matthews_r = matthews_metric.compute(references=[0,1,1,0], predictions=[1,1,1,1])

            f1_metric = evaluate.load("f1")
            f1_r = f1_metric.compute(references=[0,1,1,0], predictions=[1, 1, 1, 1])

            accuracy_metric = evaluate.load("f1")
            accuracy_r = accuracy_metric.compute(references=[0,1,1,0], predictions=[1, 1, 1, 1])
        
            hamming_r = hamming_loss(y_trueEncoded,y_predEncoded)

            time.sleep(5)
            st.success('Done!')
        
        def plot_evaluation(hamming_r, precision_r, accuracy_r, f1_r, matthews_r):
            fig = go.Figure()

            fig.add_trace(go.Bar(
                     x= hamming_r,
                     name = "hamming loss",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

            fig.add_trace(go.Bar(
                     x= precision_r,
                     name = "precision",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

            fig.add_trace(go.Bar(
                     x= accuracy_r,
                     name = "accuracy",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

            fig.add_trace(go.Bar(
                     x= f1_r,
                     name = "F1 Score",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

            fig.add_trace(go.Bar(
                     x= matthews_r,
                     name = "Matthews Correlation",
                     orientation = "h",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

            fig.update_layout(
            title_text = "Evaluation of xlm-roberta-large-xnli Language Model on English Hateful Tweets",
            title_font_size = 16,
            title_font_color = "crimson",
            title_font_family = "Gravitas One",
            xaxis=dict(
            title = "Month",
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        
        ),
            yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
            barmode='group',
            paper_bgcolor='rgb(248, 248, 255)',
            plot_bgcolor='rgb(248, 248, 255)',
            margin=dict(l=120, r=10, t=140, b=80),

            legend = dict(x=0, y=1.0, bgcolor = "LemonChiffon", bordercolor = "gold"))

            return st.plotly_chart(fig, sharing="streamlit", use_container_width=True)

        with st.container():
            plot_evaluation(hamming_r, precision_r, accuracy_r, f1_r, matthews_r)
            st.snow()

#if sidebar_option == "Kiswahili Corpus" and model == "RoBERTa-Large":


#if sidebar_option == "Kiswahili Corpus" and model == "Facebook-BART-Large":



###### English first 100 hateful tweets
# dataset logic
if option == "English Corpus (Hateful)":

    if model == "RoBERTa-Large":
        classifier = load_classifier(pipeline_one)

        eng_hateful_tweets = eng_tweets.iloc[0:101]

        def train_hate_classifier_model(df):
            scores = []
            candidate_labels = ["hateful"]
            hypothesis_template = "this sentence is {}."

            for sequence in df.text:
  # without multiclass
                score = classifier(sequence, candidate_labels, hypothesis_template)
                scores.append(score)
                df = pd.DataFrame.from_dict(scores)
                df.reset_index(drop=True)
  # convert the list entries into Series/ column values
            corpus_1 = df["labels"].apply(pd.Series)
            corpus_2 = df["scores"].apply(pd.Series)
  # rename the column names
            corpus_1.reset_index()
            corpus_1.rename(columns = {0:'label_one'}, inplace = True)
            corpus_2.rename(columns = {0:'score_one'}, inplace = True)
            corpus = pd.concat([corpus_1, corpus_2], axis=1)
  # groupby labels
            plot = corpus.groupby(["label_one"])["score_one"].mean().reset_index()
            df = plot.copy()

            return df
        
        new_plot = train_hate_classifier_model(eng_hateful_tweets)

        def draw_chart(df):

            fig = make_subplots(rows=1, cols=1, shared_yaxes=True, shared_xaxes=True)

            fig.add_trace(go.Bar(y=df["label_one"], 
                     x=df["score_one"],
                     name = "prediction on tweet",
                     orientation = "h",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")),
              1, 1)


            fig.update_layout(title = "Performance of xlm-roberta-large-xnli on English Hateful Tweets",
                  xaxis_title = "Model Score",
                  yaxis_title = "Class label",
                  coloraxis=dict(colorscale='Bluered_r'), 
                  plot_bgcolor = "white",
                  barmode = "group",
                  showlegend=False)
            return st.plotly_chart(fig, use_container_width=True)

        draw_chart(new_plot)

    # evaluate predictions

    if model == "BART-Large":
        classifier = load_classifier(pipeline_two)   

        new_plot = train_hate_classifier_model(eng_hateful_tweets)

        def draw_chart(df):

            fig = make_subplots(rows=1, cols=1, shared_yaxes=True, shared_xaxes=True)

            fig.add_trace(go.Bar(y=df["label_one"], 
                     x=df["score_one"],
                     name = "prediction on tweet",
                     orientation = "h",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")),
              1, 1)


            fig.update_layout(title = "Performance of facebook/bart-large-mnli on English Hateful Tweets",
                  xaxis_title = "Model Score",
                  yaxis_title = "Class label",
                  coloraxis=dict(colorscale='Bluered_r'), 
                  plot_bgcolor = "white",
                  barmode = "group",
                  showlegend=False)
            return st.plotly_chart(fig, use_container_width=True)

        draw_chart(new_plot)

    # evaluate predictions

if option == "English Corpus (Non-Hateful)":
    if model == "RoBERTa-Large":
        classifier = load_classifier(pipeline_one)

        eng_non_hateful_tweets = eng_tweets.iloc[100:201]

        def train_hate_classifier_model(df):
            scores = []
            candidate_labels = ["hateful"]
            hypothesis_template = "this sentence is {}."

            for sequence in df.text:
  # without multiclass
                score = classifier(sequence, candidate_labels, hypothesis_template)
                scores.append(score)
                df = pd.DataFrame.from_dict(scores)
                df.reset_index(drop=True)
  # convert the list entries into Series/ column values
            corpus_1 = df["labels"].apply(pd.Series)
            corpus_2 = df["scores"].apply(pd.Series)
  # rename the column names
            corpus_1.reset_index()
            corpus_1.rename(columns = {0:'label_one'}, inplace = True)
            corpus_2.rename(columns = {0:'score_one'}, inplace = True)
            corpus = pd.concat([corpus_1, corpus_2], axis=1)
  # groupby labels
            plot = corpus.groupby(["label_one"])["score_one"].mean().reset_index()
            df = plot.copy()

            return df
        
        new_plot = train_hate_classifier_model(eng_non_hateful_tweets)

        def draw_chart(df):

            fig = make_subplots(rows=1, cols=1, shared_yaxes=True, shared_xaxes=True)

            fig.add_trace(go.Bar(y=df["label_one"], 
                     x=df["score_one"],
                     name = "prediction on tweet",
                     orientation = "h",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")),
              1, 1)


            fig.update_layout(title = "Performance of xlm-roberta-large-xnli on Swahili Non-Hateful Tweets",
                  xaxis_title = "Model Score",
                  yaxis_title = "Class label",
                  coloraxis=dict(colorscale='Bluered_r'), 
                  plot_bgcolor = "white",
                  barmode = "group",
                  showlegend=False)
            return st.plotly_chart(fig, use_container_width=True)

        draw_chart(new_plot)

    # evaluate predictions

    if model == "BART-Large":
        classifier = load_classifier(pipeline_two)   

        new_plot = train_hate_classifier_model(eng_non_hateful_tweets)

        def draw_chart(df):

            fig = make_subplots(rows=1, cols=1, shared_yaxes=True, shared_xaxes=True)

            fig.add_trace(go.Bar(y=df["label_one"], 
                     x=df["score_one"],
                     name = "prediction on tweet",
                     orientation = "h",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")),
              1, 1)


            fig.update_layout(title = "Performance of facebook/bart-large-mnli on Swahili Non-Hateful Tweets",
                  xaxis_title = "Model Score",
                  yaxis_title = "Class label",
                  coloraxis=dict(colorscale='Bluered_r'), 
                  plot_bgcolor = "white",
                  barmode = "group",
                  showlegend=False)
            return st.plotly_chart(fig, use_container_width=True)

        draw_chart(new_plot)

    # evaluate predictions

#if option == "English Corpus (Hateful/Non-H)":


# create subset


###### Optimization  ######
@st.cache(persist=True)
def load_classifier(classifier):
    classifier = pipeline("zero-shot-classification", model= classifier)
    return classifier

@st.cache(persist=True)
def train_hate_classifier_model(df):
            scores = []
            candidate_labels = ["hateful"]
            hypothesis_template = "this sentence is {}."

            for sequence in df.text:
  # without multiclass
                score = classifier(sequence, candidate_labels, hypothesis_template)
                scores.append(score)
                df = pd.DataFrame.from_dict(scores)
                df.reset_index(drop=True)
  # convert the list entries into Series/ column values
            corpus_1 = df["labels"].apply(pd.Series)
            corpus_2 = df["scores"].apply(pd.Series)
  # rename the column names
            corpus_1.reset_index()
            corpus_1.rename(columns = {0:'label_one'}, inplace = True)
            corpus_2.rename(columns = {0:'score_one'}, inplace = True)
            corpus = pd.concat([corpus_1, corpus_2], axis=1)
  # groupby labels
            plot = corpus.groupby(["label_one"])["score_one"].mean().reset_index()
            df = plot.copy()

            return df


@st.cache(persist=True)
def train(df):

    category_map = {"hateful": "hate", "non-hateful": "loving"}

    candidate_labels = list(category_map.values())
 #candidate labels are basically the classes that the classifier will predict
    predictedCategories = []
    trueCategories = []
    for i in tqdm(range(200)):
        text = df.iloc[i,]['text']
        cat = df.iloc[i,]['label']
        cat = cat.split()
        res = classifier(text, candidate_labels)#setting multi-class as True
        labels = res['labels'] 
        scores = res['scores'] #extracting the scores associated with the labels
        res_dict = {label : score for label,score in zip(labels, scores)}
        sorted_dict = dict(sorted(res_dict.items(), key=lambda x:x[1],reverse = True)) #sorting the dictionary of labels in descending order based on their score
        categories  = []
        for i, (k,v) in enumerate(sorted_dict.items()):
            if(i > 2): #storing only the best 3 predictions
                break
            else:
                categories.append(k)
        predictedCategories.append(categories)
        trueCats = [category_map[x] for x in cat]
        trueCategories.append(trueCats)

        category_idx = {cat : i for i,cat in enumerate(category_map.values())} 
        y_trueEncoded = []
        y_predEncoded = []
        for y_true, y_pred in zip(trueCategories, predictedCategories):
            encTrue = [0] * len(category_map)
            for cat in y_true:
                idx = category_idx[cat]
                encTrue[idx] = 1
            y_trueEncoded.append(encTrue)
            encPred = [0] * len(category_map)
            for cat in y_pred:
                idx = category_idx[cat]
                encPred[idx] = 1
                y_predEncoded.append(encPred)
    return y_trueEncoded, y_predEncoded
@st.cache(persist=True)

def plot_evaluation(hamming_r, precision_r, accuracy_r, f1_r, matthews_r):
    fig = go.Figure()

    fig.add_trace(go.Bar(
                     x= hamming_r,
                     name = "hamming loss",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

    fig.add_trace(go.Bar(
                     x= precision_r,
                     name = "precision",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

    fig.add_trace(go.Bar(
                     x= accuracy_r,
                     name = "accuracy",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

    fig.add_trace(go.Bar(
                     x= f1_r,
                     name = "F1 Score",
                     orientation = "v",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

    fig.add_trace(go.Bar(
                     x= matthews_r,
                     name = "Matthews Correlation",
                     orientation = "h",
                     marker=dict(color=[2, 5, 6], coloraxis="coloraxis")))

    fig.update_layout(
            title_text = "Evaluation of xlm-roberta-large-xnli Language Model on English Hateful Tweets",
            title_font_size = 16,
            title_font_color = "crimson",
            title_font_family = "Gravitas One",
            xaxis=dict(
            title = "Month",
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        
        ),
            yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
            barmode='group',
            paper_bgcolor='rgb(248, 248, 255)',
            plot_bgcolor='rgb(248, 248, 255)',
            margin=dict(l=120, r=10, t=140, b=80),

            legend = dict(x=0, y=1.0, bgcolor = "LemonChiffon", bordercolor = "gold"))

    return st.plotly_chart(fig, sharing="streamlit", use_container_width=True)
