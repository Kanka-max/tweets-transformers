import time 
from datetime import datetime
import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="expanded",
                   page_title= "D-EQ Data App", page_icon=":sunny:" )
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import transformers
import plotly.graph_objects as go
from transformers import pipeline
from PIL import Image
image = Image.open('sunset.jpg')




import streamlit.components.v1 as components

# load corpus and split them to hateful vs non-hateful
eng_tweets = "en_200_examples.xlsx - en_200_examples.csv"
swa_tweets = "swa_tweets.csv"
eng_tweets = pd.read_csv(eng_tweets)
subset = eng_tweets[["Language","text","label"]]

swa_tweets = pd.read_csv(swa_tweets)
subset_1 = swa_tweets[["Language", "text", "label"]]


#TITLE
st.image(image, caption='Sunset by the mountains')
st.title("Transformers For Zero-Shot :sunglasses:")
st.markdown("*Huggingface* models inferred on multilingual corpus for *zero-shot* text classification")



# design side bar
st.sidebar.header("Language and Model Selection")
st.sidebar.write('The following experiment utilizes two corpus dataset: 200 English Tweets (100 labelled hateful tweets and 100 labelled non-hateful tweets); 200 Swahili Tweets (100 labelled hateful tweets and 100 labelled non-hateful tweets) :sunglasses:')

## define menu options

with st.container():

    lang = ["English", "Kiswahili", "Cross-lingual"]
    language = st.sidebar.selectbox("Select Language ", lang)
with st.container():

    mod = ["facebook/bart-large-mnli", "joeddav/xlm-roberta-large-xnli"]
    model = st.sidebar.selectbox("Choose Model ", mod)
    with st.expander("Model Documentation"):
        st.markdown("Read more on: [Bart-Large-MNLI](https://huggingface.co/facebook/bart-large-mnli)")
        st.markdown("Read more on: [Roberta-Large-XNLI](https://huggingface.co/joeddav/xlm-roberta-large-xnli)")

## create menu logics

with st.container():

    if language == "English":
        
        if model == "facebook/bart-large-mnli":

            def initialize_classifier(model):

                classifier = pipeline("zero-shot-classification",
                                      model=model)
                return classifier
            with st.spinner('Wait for it...'):
                classifier = initialize_classifier(model)
                time.sleep(5)
            st.success('Done!')
        


            hateful_tweets = subset.iloc[0:100]
            non_hateful = subset.iloc[100:200]
        # function that takes a subset and produces another subset with predictions
        # loop through the text (tweets) col
            def pick_subset(x):

                scores = []
                candidate_labels = ["hateful", "loving"]
                hypothesis_template = "this sentence is {}."
                for sequence in x.text:
        # with multiclass
                    score = classifier(sequence, candidate_labels, hypothesis_template, multi_class=True)
                    scores.append(score)
                    df = pd.DataFrame.from_dict(scores)
                    df.reset_index(drop=True)

                corpus_1 = df["labels"].apply(pd.Series)
                corpus_2 = df["scores"].apply(pd.Series)

                corpus_1.reset_index()
                corpus_1.rename(columns = {0:'label_one'}, inplace = True)
                corpus_1.rename(columns = {1:'label_two'}, inplace = True)
                corpus_2.rename(columns = {0:'score_one'}, inplace = True)
                corpus_2.rename(columns = {1:'score_two'}, inplace = True)
                corpus = pd.concat([corpus_1, corpus_2], axis=1)

                return corpus

    # hateful predictions
            corpus = pick_subset(hateful_tweets)
            hateful_preds =corpus.copy()
            plot = hateful_preds.groupby(["label_one","label_two"])["score_one","score_two"].mean().reset_index()
            hateful_preds = plot[["label_one", "score_one", "label_two", "score_two"]]
        
    #non-hateful predictions    
            corpus = pick_subset(non_hateful)
            non_hateful_preds =corpus.copy()

            plot = non_hateful_preds.groupby(["label_one","label_two"])["score_one","score_two"].mean().reset_index()
            non_hateful_preds = plot[["label_one", "score_one", "label_two", "score_two"]]

    #draw chart
            
            def draw_chart(df, df_1):


                fig = make_subplots(rows=2, cols=1, shared_yaxes=True, subplot_titles=("Hateful Tweets","Non-Hateful Tweets"))

                fig.add_trace(go.Bar(y=df["label_one"], 
                                 x=df["score_one"],
                                 name = "Prediction (Hateful Tweet)",
                                 orientation= "h",
                                 text = df["label_one"],
                                 marker=dict(color=[4, 5, 6], coloraxis="coloraxis")),
                          1, 1)

                fig.add_trace(go.Bar(y= df_1["label_one"], 
                                 x=df_1["score_one"],
                                 name = "Prediction (Non-Hateful Tweet)",
                                 text = df_1["label_one"],
                                 orientation = "h",
                                 marker=dict(color=[2, 3, 5], 
                                 coloraxis="coloraxis")),
                          2, 1)

                fig.update_layout(
                    xaxis_title = "Class labels",
                    yaxis_title = "Model Score",
                    coloraxis=dict(colorscale='Bluered_r'), 
                    showlegend=False)
            
                return st.plotly_graph(fig)
            with st.spinner('Wait for it...'):
                st.subheader("Facebook/bart-large-mnli on English Tweets")
                draw_chart(hateful_preds, non_hateful_preds)
                st.caption("A subplot comparing prediction scores on hateful vs non_hateful tweets")
                time.sleep(5)
            st.success('Done!')


## second model for English

        with st.container():

        

            if model == "joeddav/xlm-roberta-large-xnli":
                classifier = initialize_classifier(model)
                with st.spinner('Wait for it...'):
                    corpus = pick_subset(hateful_tweets)
                    hateful_preds =corpus.copy()
                    plot = hateful_preds.groupby(["label_one","label_two"])["score_one","score_two"].mean().reset_index()
                    hateful_preds = plot[["label_one", "score_one", "label_two", "score_two"]]

            #non-hateful preds
                    corpus = pick_subset(non_hateful)
                    non_hateful_preds = corpus
                    plot = non_hateful_preds.groupby(["label_one","label_two"])["score_one","score_two"].mean().reset_index()
                    non_hateful_preds = plot[["label_one", "score_one", "label_two", "score_two"]]
            
            #draw chart
                    st.subheader("xlm-roberta-large-xnli on English Tweets")
                    draw_chart(hateful_preds, non_hateful_preds)
                    st.caption("A subplot comparing prediction scores on hateful vs non_hateful tweets")
                    time.sleep(5)
                st.success('Done!')



############ KISWAHILI ##########

## create menu logics

with st.container():

    if language == "Kiswahili":
        hateful_tweets = subset_1.iloc[0:100]
        non_hateful = subset_1.iloc[100:200]

        with st.container():

            if model == "facebook/bart-large-mnli":
                classifier = initialize_classifier(model)
                with st.spinner('Wait for it...'):
                    corpus = pick_subset(hateful_tweets)
                    hateful_preds =corpus.copy()
                    plot = hateful_preds.groupby(["label_one","label_two"])["score_one","score_two"].mean().reset_index()
                    hateful_preds = plot[["label_one", "score_one", "label_two", "score_two"]]

            #non-hateful preds
                    corpus = pick_subset(non_hateful)
                    non_hateful_preds = corpus
                    plot = non_hateful_preds.groupby(["label_one","label_two"])["score_one","score_two"].mean().reset_index()
                    non_hateful_preds = plot[["label_one", "score_one", "label_two", "score_two"]]
            
            #draw chart
                    st.subheader("facebook/bart-large-mnli on English Tweets")
                    draw_chart(hateful_preds, non_hateful_preds)
                    st.caption("A subplot comparing prediction scores on hateful vs non_hateful tweets")
                    time.sleep(5)
                st.success('Done!')
        with st.container():

            if model == "joeddav/xlm-roberta-large-xnli":
                
                classifier = initialize_classifier(model)
                with st.spinner('Wait for it...'):
                    corpus = pick_subset(hateful_tweets)
                    hateful_preds =corpus.copy()
                    plot = hateful_preds.groupby(["label_one","label_two"])["score_one","score_two"].mean().reset_index()
                    hateful_preds = plot[["label_one", "score_one", "label_two", "score_two"]]

            #non-hateful preds
                    corpus = pick_subset(non_hateful)
                    non_hateful_preds = corpus
                    plot = non_hateful_preds.groupby(["label_one","label_two"])["score_one","score_two"].mean().reset_index()
                    non_hateful_preds = plot[["label_one", "score_one", "label_two", "score_two"]]
            
            #draw chart
                    draw_chart(hateful_preds, non_hateful_preds)
                    st.caption("A subplot comparing prediction scores on hateful vs non_hateful tweets")
                    time.sleep(5)
                st.success('Done!')
##### CALLBACKS

@st.cache(persist=True)
def initialize_classifier(model):

    classifier = pipeline("zero-shot-classification",
                          model=model)
    return classifier


@st.cache(persist=True)
def pick_subset(x):

    scores = []
    candidate_labels = ["hateful", "loving"]
    hypothesis_template = "this sentence is {}."
    for sequence in x.text:
        # with multiclass
        score = classifier(sequence, candidate_labels, hypothesis_template, multi_class=True)
        scores.append(score)
        df = pd.DataFrame.from_dict(scores)
        df.reset_index(drop=True)

        corpus_1 = df["labels"].apply(pd.Series)
        corpus_2 = df["scores"].apply(pd.Series)

        corpus_1.reset_index()
        corpus_1.rename(columns = {0:'label_one'}, inplace = True)
        corpus_1.rename(columns = {1:'label_two'}, inplace = True)
        corpus_2.rename(columns = {0:'score_one'}, inplace = True)
        corpus_2.rename(columns = {1:'score_two'}, inplace = True)
        corpus = pd.concat([corpus_1, corpus_2], axis=1)

        return corpus


@st.cache(persist=True)
def draw_chart(df, df_1):


    fig = make_subplots(rows=2, cols=1, shared_yaxes=True, subplot_titles=("Hateful Tweets","Non-Hateful Tweets"))

    fig.add_trace(go.Bar(y=df["label_one"], 
                                 x=df["score_one"],
                                 name = "Prediction (Hateful Tweet)",
                                 orientation= "h",
                                 text = df["label_one"],
                                 marker=dict(color=[4, 5, 6], coloraxis="coloraxis")),
                          1, 1)

    fig.add_trace(go.Bar(y= df_1["label_one"], 
                                 x=df_1["score_one"],
                                 name = "Prediction (Non-Hateful Tweet)",
                                 text = df_1["label_one"],
                                 orientation = "h",
                                 marker=dict(color=[2, 3, 5], 
                                 coloraxis="coloraxis")),
                          2, 1)

    fig.update_layout(title = "Performance of facebook/bart-large-mnli on English Tweets",
                    xaxis_title = "Class labels",
                    yaxis_title = "Model Score",
                    coloraxis=dict(colorscale='Bluered_r'), 
                    showlegend=False)
            
    return st.plotly_graph(fig)