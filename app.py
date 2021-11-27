#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st

#import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from urllib.request import urlopen, Request
import re 
from bs4 import BeautifulSoup
from imageio import imread
import cv2

import tensorflow as tf
import tensorflow_addons as tfa
from transformers import BertTokenizer
from transformers import TFAutoModel
from tensorflow.keras.layers.experimental import preprocessing

import unicodedata
import joblib

#function used for scrapping:
def get_info_from_amazon(link):

    #On retire la fin du lien, qui peut contenir des caractères non ascii
    #les liens amazon peuvent etre résumés en : https://www.amazon.fr/dp/[code produit]
    r1 = re.compile(r"dp/+[a-zA-Zéà0-9]+/")
    code_product = r1.findall(link)[0]

    #domaine amazon.fr ou .com
    r2 = re.compile(r"https?://[a-zA-Zéà0-9.]+/")
    domain = r2.findall(link)[0]

    new_link = domain + code_product
    
    #on récupère les infos de la page
    headers = ({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36',
            'Connection':'keep-alive'})
    req = Request(url = new_link, headers=headers)
    page = urlopen(req)
    soup = BeautifulSoup(page, 'html.parser')
    
    #Recuperation du titre
    titre = soup.findAll(name = 'span', attrs = {'id':  'productTitle'})
    if titre:
        designation = titre[0].text.strip()
    else:
        designation = 'no designation found'
    
    #Recuperation de la description
    #description near image
    desc1 = soup.findAll(name = 'div', attrs = {'id': 'iframeContent'})
    #description in more details
    desc2 = soup.findAll(name = 'div', attrs = {'id': 'feature-bullets'})
    #description below
    desc3 = soup.findAll(name = 'div', attrs = {'id': 'productDescription'})
    
    if desc1:
        description = desc1[0].text.strip()   
    elif desc2:
        description = desc2[0].text.strip()  
    elif desc3:
        description = desc3[0].text.strip() 
    else:
        description = 'no description found'
        
    #Recuperation de l'image
    img1 = soup.findAll(name = 'div', attrs= {'id' : 'imgTagWrapperId'})
    img2 = soup.findAll(name = 'div', attrs= {'id' :  'main-image-container'})
    img3 = soup.findAll(name = 'div', attrs= {'class' : 'image-wrapper'})
    
    if img1:
        src_img = img1[0].img['src']
        image = imread(src_img)
    elif img2:
        src_img = img2[0].img['src']
        image = imread(src_img)
    elif img3:
        src_img = img3[0].img['src']
        image = imread(src_img)
    else:
        image = []
    
    return [designation,description,image]

def get_random_rakuten():
    rand = np.random.randint(len(df))
    designation = df.iloc[rand,0]
    description = df.iloc[rand,1]
    path_img = './image/'+df.iloc[rand,2]
    return [designation, description,path_img] 

st.title("Prediction de la typologie de produit")

#Table d'annonce scrappées sur rakuten
df= pd.read_csv('./article_rakuten_scraped.csv', sep='\t')
#Function for prediction

def prediction():

    #A METTRE A JOUR AVEC LES BONS PATHS LOCAUX
    MODEL_TEXT = './models/bert_model.hdf5'
    
    CATEGORIES_PATH = './cat.csv'
    
    IMG_SIZE = 456

    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(w):
        w = unicode_to_ascii(w.lower().strip())
        return w

    @st.cache(allow_output_mutation=True)
    def load_model_text():
        model = TFAutoModel.from_pretrained('bert-base-multilingual-cased')
        f1w = tfa.metrics.F1Score(num_classes = 27, average = 'weighted')
        model = tf.keras.models.load_model(MODEL_TEXT,
                                   custom_objects= {'f1_score': f1w})
        return model

    @st.cache(allow_output_mutation=True)
    def prep_data(text):

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        tokens = tokenizer.encode_plus(text, max_length=512, truncation=True,
                                    padding='max_length', add_special_tokens=True,
                                    return_tensors='tf')
        return{'input_ids': tf.cast(tokens['input_ids'], tf.float64), 
                'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)
                }

    categories = pd.read_csv(CATEGORIES_PATH, sep=',')

    #Collect text info
    if description != 'description not found':
        sentence = designation+' '+description
    else:
        sentence=designation

    sentence = BeautifulSoup(sentence, features="lxml").get_text()
    #sentence = BeautifulSoup(sentence, features="html.parser").get_text()
    sentence = preprocess_sentence(sentence)
    sentence = prep_data(sentence)

    model_text = load_model_text()
    if len(sentence) > 0:
        probas_text = model_text.predict(sentence)
        response_text = np.argmax(probas_text[0])
        res_text = categories[(categories['encoded_target'] == response_text)]
        st.title("MODELE TEXTE")
        st.markdown("**La catégorie calculée par le modèle est : **")
        st.text(res_text['prdtypecode'].values[0])
        st.text(res_text['libelle'].values[0])

#Page
st.header('1/ Annonce à analyser:')
option = st.selectbox("Source de l'annonce" , 
                      ("Choix d'un url amazon", "Annonce aléatoire Rakuten"))

#Cas 1: amazon via url

if option == "Choix d'un url amazon":
    
    st.header('Entrez le lien du produit amazon:')
    link = st.text_input('Lien')

    if link:
        info_link = get_info_from_amazon(link)
        designation = info_link[0]
        description = info_link[1]
        image = info_link[2]
        
        st.header("Informations de l'annonce")
        st.markdown('* Designation (titre) : ')
        st.markdown(designation)
        st.markdown('* Description (si renseigné): ')
        st.markdown(description)
        
        if len(image)>1:
            st.markdown('* Image')
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(1,1,1)
            ax.imshow(image)
            ax.axis('off')
            st.write(fig)
            
            #Prediction
            st.header('2/Predire la catégorie du produit:')
            prediction()
            
        else:
            st.write('no image found')
        


    
if option == "Annonce aléatoire Rakuten":

    go_rak = st.button('générer une annonce rakuten')
    if go_rak:
        designation, description, IMAGE_PATH = get_random_rakuten()
        image = imread(IMAGE_PATH)
        
        
        st.header("Informations de l'annonce")
        st.markdown('* Designation (titre) : ')
        st.markdown(designation)
        st.markdown('* Description (si renseigné): ')
        st.markdown(description)
        
        if image.any():
            st.markdown('* Image')
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(1,1,1)
            ax.imshow(image)
            ax.axis('off')
            st.write(fig)
        else:
            st.write('no image found')   
        
        #Prediction
        st.header('2/Predire la catégorie du produit:')
        prediction()



              