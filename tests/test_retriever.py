from ragbot.retriever import HybridRetriever, reciprocal_rank_fusion, tokenize


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is error CD-1042?") == ["error", "cd", "1042"]


def test_rrf_rewards_documents_ranked_well_in_both_lists():
    vector = ["a", "b", "c"]
    keyword = ["b", "d", "e"]
    fused = [doc_id for doc_id, _ in reciprocal_rank_fusion([vector, keyword])]
    assert fused[0] == "b"  # found by both searches beats rank 1 in only one
    assert set(fused) == {"a", "b", "c", "d", "e"}


def test_rrf_score_formula():
    ((doc_id, score),) = reciprocal_rank_fusion([["x"]], k=60)
    assert doc_id == "x"
    assert score == 1 / 61


def test_store_returns_cosine_similarities(store):
    results = store.search("parental leave weeks", 3)
    assert len(results) == 3
    similarities = [score for _, score in results]
    assert similarities == sorted(similarities, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in similarities)


def test_keyword_search_finds_exact_codes(store):
    retriever = HybridRetriever(store)
    top = retriever.keyword_search("error CD-1042", 1)
    assert "CD-1042" in retriever.chunks_by_id[top[0]].text


def test_hybrid_retrieve_returns_top_k_chunks(store):
    retriever = HybridRetriever(store, candidates=10)
    results = retriever.retrieve("How many weeks of parental leave?", top_k=3)
    assert len(results) == 3
    assert any("16 weeks" in r.chunk.text for r in results)


def test_empty_store_returns_nothing(tmp_path):
    from ragbot.store import VectorStore
    from tests.conftest import HashEmbedder

    empty = VectorStore(str(tmp_path / "empty"), "empty", embedder=HashEmbedder())
    assert HybridRetriever(empty).retrieve("anything") == []