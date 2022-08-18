from krwordrank.word import KRWordRank


def _make_KRWordRank_voca(documents, min_count, max_length):
    vocabulary = {}

    counter = {}
    for doc in documents:

        for token in doc.split():
            len_token = len(token)
            counter[(token, 'L')] = counter.get((token, 'L'), 0) + 1

            for e in range(1, min(len(token), max_length)):
                if (len_token - e) > max_length:
                    continue

                l_sub = (token[:e], 'L')
                r_sub = (token[e:], 'R')
                counter[l_sub] = counter.get(l_sub, 0) + 1
                counter[r_sub] = counter.get(r_sub, 0) + 1

    counter = {token: freq for token, freq in counter.items() if freq >= min_count}
    for token, _ in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        vocabulary[token] = len(vocabulary)

    return counter


def execute_KRWordRank(documents, min_count=5, max_length=10, beta=0.85, max_iter=10, num_words=100):
    vocabulary = _make_KRWordRank_voca(documents, min_count)

    krword_rank = KRWordRank(min_count, max_length, verbose=True)
    # keywords, rank, graph = krword_rank.extract(docs=documents, beta=beta, max_iter=max_iter, num_keywords=num_words,
    #                                             vocabulary=vocabulary)

    keywords, rank, graph = krword_rank.extract(docs=documents, beta=beta, max_iter=max_iter, num_keywords=num_words)

    print(krword_rank.vocabulary)

    return keywords
