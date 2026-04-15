import re
from sklearn.feature_extraction.text import TfidfVectorizer

def lyrics_features(lyrics):
    # get clean lyrics
    cleaned = [clean_lyrics(l) for l in lyrics]

    # vectorise
    vec = TfidfVectorizer(max_features=500, stop_words='english')

    return vec.fit_transform(cleaned).toarray()

def clean_lyrics(text):
    text = text.lower()

    # remove [VERSE 1], [CHORUS] etc.
    text = re.sub(r'\[.*?\]', ' ', text)

    # remove (Verse 1), (Chorus) etc.
    text = re.sub(r'\(.*?\)', ' ', text)

    # remove newline and escape chars
    text = text.replace('\n', ' ').replace('\r', ' ')

    # remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
