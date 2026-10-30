import pytest

from src.reviews import latest_reviews, review_history, save_review


def test_review_persists_and_retains_audit(tmp_path):
    path = tmp_path / "reviews.sqlite3"
    save_review("ACC1", "Confirmed", "AK", "Verified source", "v1", path)
    save_review("ACC1", "Dismissed", "AK", "Corrected stale bureau", "v1", path)
    save_review("ACC1", "Needs investigation", "BK", "New snapshot", "v2", path)
    history = review_history(path)
    assert len(history) == 3
    assert latest_reviews(path, "v1").iloc[0].decision == "Dismissed"
    assert latest_reviews(path, "v2").iloc[0].decision == "Needs investigation"


def test_review_parameters_and_validation(tmp_path):
    path = tmp_path / "reviews.sqlite3"
    malicious = "x'); DROP TABLE reviews; --"
    save_review("ACC1", "Confirmed", "AK", malicious, "v1", path)
    assert review_history(path).iloc[0].note == malicious
    with pytest.raises(ValueError):
        save_review("ACC1", "Approved", "AK", "note", "v1", path)
    with pytest.raises(ValueError):
        save_review("ACC1", "Dismissed", "", "", "v1", path)
