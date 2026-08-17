import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from string import punctuation
from nltk.stem import SnowballStemmer

nltk.download("stopwords", quiet=True)

STOP_WORDS = stopwords.words('english')
PUNCTUATION = list(punctuation)
NOISE = STOP_WORDS + PUNCTUATION + ['da']

tweet_tokenizer = TweetTokenizer()
stemmer = SnowballStemmer('english')

def custom_new_tokenizer(text:str):
    '''
    Приводит текст к нижнему регистру, токенизирует с помощью TweetTokenizer, 
    удаляет стоп-слова, пунктуацию, одиночные символы с ord >= 128, отдельно стоящие числа и ссылки на t.co.
    Использует SnowballStemmer для стемминга
    '''
    text = text.lower()
    tokens = tweet_tokenizer.tokenize(text)
    cleaned = []
    for token in tokens:
        if token in NOISE:
            continue
        if len(token) == 1 and ord(token) >= 128:
            continue
        if token.startswith("https://t.co"):
            continue
        if token.isdigit():
            continue
        cleaned.append(token)
    return [stemmer.stem(token) for token in cleaned]