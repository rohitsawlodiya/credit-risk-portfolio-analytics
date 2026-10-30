import json

from streamlit.testing.v1 import AppTest

from src.config import ROOT
from src.reviews import review_history
from tests.test_assistant import plan


def start_app(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("REVIEW_DB_PATH", str(tmp_path / "reviews.sqlite3"))
    return AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()


def test_all_views_and_empty_filter(monkeypatch, tmp_path):
    app = start_app(monkeypatch, tmp_path)
    assert not app.exception
    assert app.metric[0].value == "14,685"
    for view in ["Account explorer", "Anomaly review", "Ask the Portfolio", "Model & data", "Portfolio"]:
        app.radio[0].set_value(view).run()
        assert not app.exception, f"Failed view: {view}"
    app.multiselect(key="filter_region").set_value([]).run()
    assert app.metric[0].value == "0"
    assert not app.exception


def test_review_form_saves_to_disk(monkeypatch, tmp_path):
    app = start_app(monkeypatch, tmp_path)
    app.radio[0].set_value("Anomaly review").run()
    app.text_input[0].set_value("Test analyst")
    app.text_area[0].set_value("Inspected repayment and bureau data")
    app.button[0].click().run()
    assert not app.exception
    saved = review_history(tmp_path / "reviews.sqlite3")
    assert len(saved) == 1 and saved.iloc[0].decision == "Confirmed"


def test_llm_page_mocked_end_to_end(monkeypatch, tmp_path):
    from tests.test_assistant import mock_client
    import src.assistant as module
    app = start_app(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-ui-test-key")
    monkeypatch.setattr(module, "OpenAI", lambda **_: mock_client(plan()))
    app.radio[0].set_value("Ask the Portfolio").run()
    app.text_input[0].set_value("Which loan type has the highest default rate?")
    app.button[0].click().run()
    assert not app.exception
    assert any("highest observed default rate" in x.value for x in app.info)
    app.multiselect(key="filter_region").set_value(["North"]).run()
    assert any("Filters changed" in x.value for x in app.info)
