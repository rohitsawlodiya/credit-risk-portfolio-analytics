import pytest
from pypdf import PdfReader

from src.config import ROOT


@pytest.mark.parametrize("name", ["architecture", "solution_overview"])
def test_required_documents_are_exactly_one_page(name, report):
    reader = PdfReader(ROOT / "docs" / f"{name}.pdf")
    assert len(reader.pages) == 1
    assert len(reader.pages[0].extract_text()) > 1000
    if name == "solution_overview":
        content = reader.pages[0].extract_text()
        for model in report["model"]["models"].values():
            for metric in ["auc", "ks", "gini"]:
                assert f"{model['test'][metric]:.4f}" in content
