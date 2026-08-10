from pipeline.rag_engine import build_rag_query


def test_build_rag_query_includes_key_fields():
    query = build_rag_query("MSV", 31.4, 3)
    assert "MSV" in query
    assert "31%" in query
    assert "grade 3" in query
    assert "Philippines" in query


def test_build_rag_query_rounds_severity():
    query = build_rag_query("MLN", 58.6, 4)
    assert "59%" in query  # :.0f rounds 58.6 up
