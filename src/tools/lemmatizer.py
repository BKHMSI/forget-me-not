from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from num2words import num2words

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    elif treebank_tag.startswith('CD'):
        return wordnet.NOUN
    else:
        return "remove"

def get_lemmatized(sentence):
    lemma = WordNetLemmatizer()
    sentence = sentence.split()
    sentence = [word for word in sentence if word not in stopwords.words('english')];
    tags = pos_tag(sentence)
    result =list()
    for index, word in enumerate(sentence):
        pos =get_wordnet_pos(tags[index][1])
        if pos != "remove":
            word = word.lower()
            word ="".join(c for c in word if c not in (';', ':', ',', '.', '!', '?'))
            if word.isdigit():
                word = num2words(int(word))
                word = word.split()
                word = " ".join(part for part in word if part not in stopwords.words('english'))
                word = word.split("-")
                word = " ".join(part for part in word)
            result.append(lemma.lemmatize(word,pos))
    return(' '.join(result))