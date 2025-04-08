from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def extract_features(texts, method='high_frequency', **kwargs):
    if method == 'high_frequency':
        vectorizer = CountVectorizer(**kwargs)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(**kwargs)
    else:
        raise ValueError("Invalid method. Choose 'high_frequency' or 'tfidf'.")

    features = vectorizer.fit_transform(texts)
    return features, vectorizer


if __name__ == "__main__":
    texts = [
        "This is a sample document.",
        "Another document with different content.",
        "A document with similar content to the first one."
    ]

    features_high_freq, vectorizer_high_freq = extract_features(texts, method='high_frequency')
    print("High Frequency Features:\n", features_high_freq.toarray())
    print("Feature Names:", vectorizer_high_freq.get_feature_names_out())

    features_tfidf, vectorizer_tfidf = extract_features(texts, method='tfidf')
    print("\nTF-IDF Features:\n", features_tfidf.toarray())
    print("Feature Names:", vectorizer_tfidf.get_feature_names_out())
