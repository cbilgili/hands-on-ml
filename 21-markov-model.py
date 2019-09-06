#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 22:53:11 2017

@author: cbilgili
"""

import random

class Dictogram(dict):
    def __init__(self, iterable=None):
        """Initialize this histogram as a new dict; update with given items"""
        super(Dictogram, self).__init__()
        self.types = 0  # the number of distinct item types in this histogram
        self.tokens = 0  # the total count of all item tokens in this histogram
        if iterable:
            self.update(iterable)

    def update(self, iterable):
        """Update this histogram with the items in the given iterable"""
        for item in iterable:
            if item in self:
                self[item] += 1
                self.tokens += 1
            else:
                self[item] = 1
                self.types += 1
                self.tokens += 1

    def count(self, item):
        """Return the count of the given item in this histogram, or 0"""
        if item in self:
            return self[item]
        return 0

    def return_random_word(self):
        # Another way:  Should test: random.choice(histogram.keys())
        random_key = random.sample(self, 1)
        return random_key[0]

    def return_weighted_random_word(self):
        # Step 1: Generate random number between 0 and total count - 1
        random_int = random.randint(0, self.tokens-1)
        index = 0
        list_of_keys = list(self.keys())
        # print 'the random index is:', random_int
        for i in range(0, self.types):
            index += self[list_of_keys[i]]
            # print index
            if(index > random_int):
                # print list_of_keys[i]
                return list_of_keys[i]
            
#from histograms import Dictogram

def make_markov_model(data):
    markov_model = dict()
    data = data.split(' ')

    for i in range(0, len(data)-1):
        if data[i] in markov_model:
            # We have to just append to the existing histogram
            markov_model[data[i]].update([data[i+1]])
        else:
            markov_model[data[i]] = Dictogram([data[i+1]])
    return markov_model

def make_higher_order_markov_model(order, data):
    markov_model = dict()
    data = data.split(' ')

    for i in range(0, len(data)-order):
        # Create the window
        window = tuple(data[i: i+order])
        # Add to the dictionary
        if window in markov_model:
            # We have to just append to the existing Dictogram
            markov_model[window].update([data[i+order]])
        else:
            markov_model[window] = Dictogram([data[i+order]])
    return markov_model

import random
from collections import deque
import re
import glob
import json


def generate_random_start(model):
    # To just generate any starting word uncomment line:
    return random.choice(list(model.keys()))
                
    # To generate a "valid" starting word use:
    # Valid starting words are words that started a sentence in the corpus
#    if 'END' in model:
#        seed_word = 'END'
#        while seed_word == 'END':
#            seed_word = model['END'].return_weighted_random_word()
#        return seed_word
#    return random.choice(list(model.keys()))


def generate_random_sentence(length, markov_model):
    current_word = generate_random_start(markov_model)
    sentence = [current_word]
    for i in range(0, length):
        current_dictogram = markov_model[current_word]
        random_weighted_word = current_dictogram.return_weighted_random_word()
        current_word = random_weighted_word
        sentence.append(current_word)
    sentence[0] = sentence[0].capitalize()
    return ' '.join(sentence) + '.'
    return sentence

def clean_text(text):
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    text = text.replace(',', ' ')
    text = text.replace("'", ' ')
    text = text.replace("`", ' ')
    text = text.replace("’", ' ')
    return text.strip()
    
# # https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/
# # clean a list of lines
# def clean_lines(lines):
#     cleaned = list()
#     # prepare regex for char filtering
#     re_print = re.compile('[^%s]' % re.escape(string.printable))
#     # prepare translation table for removing punctuation
#     table = str.maketrans('', '', string.punctuation)
#     for line in lines:
#         # normalize unicode characters
#         line = normalize('NFD', line).encode('ascii', 'ignore')
#         line = line.decode('UTF-8')
#         # tokenize on white space
#         line = line.split()
#         # convert to lower case
#         line = [word.lower() for word in line]
#         # remove punctuation from each token
#         line = [word.translate(table) for word in line]
#         # remove non-printable chars form each token
#         line = [re_print.sub('', w) for w in line]
#         # remove tokens with numbers in them
#         line = [word for word in line if word.isalpha()]
#         # store as string
#         cleaned.append(' '.join(line))
#     return cleaned
    
#initial_sentence = "Today you are you. That is truer than true. There is no one alive who is you-er than you. You have brains in your head. You have feet in your shoes. You can steer yourself any direction you choose. You’re on your own. The more that you read, the more things you will know. The more that you learn, the more places you’ll go. Think left and think right and think low and think high. Oh, the thinks you can think up if only you try."
initial_sentence = "Korkma, sönmez bu şafaklarda yüzen al sancak; Sönmeden yurdumun üstünde tüten en son ocak. O benim milletimin yıldızıdır, parlayacak; O benimdir, o benim milletimindir ancak. Çatma, kurban olayım, çehrene ey nazlı hilal! Kahraman ırkıma bir gül... Ne bu şiddet, bu celal? Sana olmaz dökülen kanlarımız sonra helal; Hakkıdır, Hakk'a tapan, milletimin istiklal. Ben ezelden beridir hür yaşadım, hür yaşarım. Hangi çılgın bana zincir vuracakmış? Şaşarım! Kükremiş sel gibiyim: Bendimi çiğner, aşarım; Yırtarım dağları, enginlere sığmam taşarım. Garb'ın afakını sarmışsa çelik zırhlı duvar; Benim iman dolu göğsüm gibi serhaddim var. Ulusun, korkma! Nasıl böyle bir imanı boğar, ''Medeniyet!'' dediğin tek dişi kalmış canavar? Arkadaş! Yurduma alçakları uğratma sakın; Siper et gövdeni, dursun bu hayasızca akın. Doğacaktır sana va'dettiği günler Hakk'ın... Kim bilir, belki yarın, belki yarından da yakın. Bastığın yerleri ''toprak!'' diyerek geçme, tanı! Düşün altındaki binlerce kefensiz yatanı. Sen şehid oğlusun, incitme, yazıktır, atanı: Verme, dünyaları alsan da, bu cennet vatanı. Kim bu cennet vatanın uğruna olmaz ki feda? Şüheda fışkıracak toprağı sıksan, şüheda! Canı, cananı, bütün varımı alsın da Huda, Etmesin tek vatanımdan beni dünyada cüda. Ruhumun senden İlahi şudur ancak emeli: Değmesin ma'bedimin göğsüne na-mahrem eli; Bu ezanlar -- ki şehadetleri dinin temeli -- Ebedi, yurdumun üstünde benim inlemeli. O zaman vecd ile bin secde eder -- varsa -- taşım; Her cerihamda, İlahi, boşanıp kanlı yaşım, Fışkırır ruh-i mücerred gibi yerden na'şım! O zaman yükselerek Arş'a değer, belki, başım. Dalgalan sen de şafaklar gibi ey şanlı hilal! Olsun artık dökülen kanlarımın hepsi helal. Ebediyyen sana yok, ırkıma yok izmihlal: Hakkıdır, hür yaşamış, bayrağımın hürriyet; Hakkıdır, Hakk'a tapan, milletimin istiklal."
model = make_markov_model(initial_sentence)
new_sentence = generate_random_sentence(1, model)



PATH = '/Users/cbilgili/Sites/labs/python/spider/data'
files = glob.glob(PATH + '/**/*.json', recursive=True)
files = [x for x in files if not "spider_default" in x]


dictogram_model = Dictogram()

for file in files:
    data = json.load(open(file))
    texts = [clean_text(text[0]) for text in [text['text'] for text in data["texts"]]]
    splitted_texts = [text.split(' ') for text in texts]
    for text in splitted_texts:
        dictogram_model.update(text)

import simplejson as json
import os
with open(os.path.join('/Users/cbilgili/Desktop', 'data.json'), 'w') as f:
    f.write(json.dumps(dictogram_model))
