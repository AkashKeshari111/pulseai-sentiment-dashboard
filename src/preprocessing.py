"""NLP preprocessing pipeline.

Two cleaning profiles are exposed, because classical and neural models want
very different inputs:

``clean_for_classical``
    Aggressive normalisation (lowercase, de-punctuate, stopword removal,
    suffix stripping). TF-IDF has no notion of morphology, so collapsing
    surface forms is what makes the bag-of-words representation work.

``clean_for_transformer``
    Light normalisation only (strip URLs/HTML/handles, fix whitespace and
    character elongation). DistilBERT's WordPiece vocabulary already handles
    casing, punctuation and sub-words - stripping them destroys signal such as
    "NOT worth it" or "great!!!".

The module is dependency-free on purpose: no NLTK/spaCy downloads are needed,
so the notebook, the API container and CI all behave identically offline.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Static resources
# ---------------------------------------------------------------------------

#: Standard English stopwords, minus negations and intensifiers. Removing
#: "not", "no", "never", "very" is a classic sentiment-analysis bug: it flips
#: "not good" into "good".
_SENTIMENT_CRITICAL = {
    "not", "no", "nor", "never", "none", "cannot", "cant", "wont", "dont",
    "didnt", "doesnt", "isnt", "wasnt", "arent", "werent", "hasnt", "havent",
    "hadnt", "shouldnt", "wouldnt", "couldnt", "very", "too", "but", "however",
    "although", "though", "against", "only", "just", "more", "most", "least",
}

_BASE_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "my",
    "myself", "of", "off", "on", "once", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to",
    "under", "until", "up", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "yourself", "yourselves", "s", "t", "ll", "re", "ve",
    "d", "m", "o", "y", "im", "ive", "id", "youre", "theyre", "hes", "shes",
    "thats", "whats", "lets", "get", "got", "one", "also", "us", "u",
}

STOPWORDS = _BASE_STOPWORDS - _SENTIMENT_CRITICAL

#: Keyword extraction has the opposite requirement to classification. "not" and
#: "very" are essential *features* for a model, but they are not *topics* - a
#: word cloud led by "not, but, just" tells an analyst nothing. So the keyword
#: stopword list keeps the negations and adds the usual conversational filler.
_KEYWORD_FILLER = {
    "really", "actually", "definitely", "probably", "maybe", "much", "many",
    "even", "still", "back", "way", "well", "want", "wanted", "know", "think",
    "thought", "said", "say", "told", "went", "came", "come", "going", "take",
    "took", "make", "made", "give", "gave", "put", "see", "saw", "look",
    "looked", "need", "needed", "use", "used", "time", "times", "day", "days",
    "thing", "things", "something", "anything", "everything", "nothing",
    "someone", "anyone", "everyone", "always", "never", "ever", "another",
    "every", "since", "around", "though", "like", "liked", "first", "last",
    "next", "two", "three", "lot", "bit", "little", "big", "new", "old",
    "place", "people", "person", "guy", "guys", "asked", "gets", "getting",
}
KEYWORD_STOPWORDS = _BASE_STOPWORDS | _SENTIMENT_CRITICAL | _KEYWORD_FILLER

#: Expanded before tokenisation so "don't" survives punctuation stripping as
#: the negation token "not" instead of dissolving into "don" + "t".
CONTRACTIONS = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it's": "it is", "let's": "let us",
    "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what's": "what is", "where's": "where is",
    "who's": "who is", "won't": "will not", "wouldn't": "would not",
    "you'd": "you would", "you'll": "you will", "you're": "you are",
    "you've": "you have", "y'all": "you all",
}

_ESCAPE_RE = re.compile(r"\\+[ntr]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_HANDLE_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")
_ELONGATION_RE = re.compile(r"(.)\1{2,}")
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b", re.IGNORECASE
)
_REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
_TOKEN_RE = re.compile(r"[a-z][a-z'-]*")


# ---------------------------------------------------------------------------
# Primitive steps
# ---------------------------------------------------------------------------

def normalise_unicode(text: str) -> str:
    """Fold accents and exotic look-alikes down to plain ASCII-ish text."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def strip_noise(text: str) -> str:
    """Remove artefacts that carry no sentiment: URLs, HTML, emails, handles."""
    # Several public review corpora store newlines as the two characters
    # backslash-n rather than a real newline. Left alone, the "n" fuses with
    # the next word and the vocabulary fills with ghosts like "nthe".
    text = _ESCAPE_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _HANDLE_RE.sub(" ", text)
    # "#terrible" -> "terrible": the word inside a hashtag is real signal.
    text = _HASHTAG_RE.sub(r"\1", text)
    return text


def expand_contractions(text: str) -> str:
    return _CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0).lower()], text)


def squash_elongation(text: str) -> str:
    """``sooooo goooood!!!!!`` becomes ``soo goood!!``.

    Keeps a doubled character/punctuation as an emphasis marker while
    collapsing the long tail that would otherwise explode the vocabulary.
    """
    text = _ELONGATION_RE.sub(r"\1\1", text)
    return _REPEAT_PUNCT_RE.sub(r"\1\1", text)


def simple_stem(token: str) -> str:
    """A conservative suffix stripper (Porter-lite).

    Full Porter stemming over-truncates short sentiment words ("worse" becomes
    "wors"); this keeps tokens of at least 4 characters and only strips the
    highest frequency inflectional suffixes, which is what TF-IDF actually
    benefits from.
    """
    if len(token) <= 4:
        return token
    for suffix in ("ingly", "edly", "ing", "ers", "ies", "ied", "ly", "es", "ed", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            if suffix in {"ies", "ied"}:
                return stem + "y"
            # "shipping" -> "ship", not "shipp"
            if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                stem = stem[:-1]
            return stem
    return token


# ---------------------------------------------------------------------------
# Public cleaning profiles
# ---------------------------------------------------------------------------

def clean_for_transformer(text: str) -> str:
    """Light cleaning for DistilBERT: preserve case, punctuation and negation."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = normalise_unicode(text)
    text = strip_noise(text)
    text = squash_elongation(text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()


def clean_for_classical(text: str, stem: bool = True) -> str:
    """Aggressive cleaning for TF-IDF: lowercase bag of normalised tokens."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = normalise_unicode(text).lower()
    text = strip_noise(text)
    text = expand_contractions(text)
    text = squash_elongation(text)
    text = _NUMBER_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)

    tokens = [tok for tok in text.split() if len(tok) > 1 and tok not in STOPWORDS]
    if stem:
        tokens = [simple_stem(tok) for tok in tokens]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, used by the keyword/issue analytics.

    Noise stripping runs first so URL fragments never leak into the token
    stream as fake keywords ("https", "com", "co").
    """
    text = squash_elongation(strip_noise(normalise_unicode(text or "")))
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Issue analytics - powers the dashboard's "what are customers complaining
# about" panel.
# ---------------------------------------------------------------------------

#: Keyword -> business issue category. Deliberately transparent and auditable:
#: an ops team can extend this list without retraining anything.
ISSUE_TAXONOMY: dict[str, tuple[str, ...]] = {
    "Delivery & Logistics": (
        "delivery", "shipping", "shipped", "courier", "late", "delay", "delayed",
        "arrived", "arrive", "tracking", "package", "parcel", "dispatch",
    ),
    "Product Quality": (
        "quality", "broken", "damaged", "defective", "cheap", "flimsy", "faulty",
        "malfunction", "worn", "torn", "leak", "crack", "sturdy", "durable",
    ),
    "Customer Support": (
        "support", "service", "staff", "agent", "rude", "helpful", "unhelpful",
        "response", "reply", "ticket", "manager", "representative", "helpline",
    ),
    "Pricing & Value": (
        "price", "expensive", "overpriced", "cost", "costly", "value", "money",
        "refund", "charge", "charged", "billing", "fee", "worth", "cheaper",
    ),
    "Usability & UX": (
        "app", "website", "interface", "confusing", "crash", "crashes", "bug",
        "login", "loading", "navigate", "checkout", "cart", "error", "glitch",
    ),
    "Food & Taste": (
        "food", "taste", "tasty", "bland", "flavor", "flavour", "fresh", "stale",
        "portion", "menu", "meal", "dish", "sauce", "cooked", "delicious",
    ),
    "Ambience & Facilities": (
        "clean", "dirty", "noisy", "crowded", "atmosphere", "ambience", "seating",
        "parking", "restroom", "decor", "music", "spacious",
    ),
    "Wait Time": (
        "wait", "waited", "waiting", "queue", "slow", "forever", "quick",
        "prompt", "instantly", "immediately",
    ),
}

_KEYWORD_TO_CATEGORY: dict[str, str] = {
    keyword: category
    for category, keywords in ISSUE_TAXONOMY.items()
    for keyword in keywords
}


def detect_issue_categories(text: str, max_categories: int = 3) -> list[str]:
    """Tag a piece of feedback with the business areas it talks about.

    Returns categories ordered by how many distinct matching keywords the text
    contains, so the dominant topic comes first.
    """
    tokens = set(tokenize(text or ""))
    hits: Counter[str] = Counter()
    for token in tokens:
        category = _KEYWORD_TO_CATEGORY.get(token) or _KEYWORD_TO_CATEGORY.get(
            simple_stem(token)
        )
        if category:
            hits[category] += 1
    return [category for category, _ in hits.most_common(max_categories)]


def extract_keywords(
    texts: Iterable[str], top_n: int = 25, min_length: int = 3
) -> list[tuple[str, int]]:
    """Frequency ranked content words across a corpus (for the word cloud).

    Uses :data:`KEYWORD_STOPWORDS` rather than :data:`STOPWORDS`: this function
    answers "what are customers talking about", and function words are noise
    for that question even though they are signal for the classifier.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        for token in tokenize(text or ""):
            token = token.strip("'-")
            if "'" in token:
                # "don't" / "it's" - the contraction stem is not a useful topic.
                continue
            if len(token) >= min_length and token not in KEYWORD_STOPWORDS:
                counter[token] += 1
    return counter.most_common(top_n)


__all__ = [
    "KEYWORD_STOPWORDS",
    "STOPWORDS",
    "ISSUE_TAXONOMY",
    "clean_for_classical",
    "clean_for_transformer",
    "detect_issue_categories",
    "expand_contractions",
    "extract_keywords",
    "simple_stem",
    "tokenize",
]
