import json
import os

import httpx
import numpy as np
from openai import OpenAI
import pytest
from pydantic import ValidationError

from src.assistant import AssistantUnavailable, QueryPlan, ask_portfolio, execute_plan, make_plan


def plan(**overrides):
    return {"action": "query", "decline_reason": "none", "metric": "default_rate",
            "group_by": ["loan_type"], "filters": [], "direction": "highest", "limit": 1,
            "min_accounts": 1, **overrides}


def test_exact_grounded_answer(portfolio):
    result = execute_plan(portfolio, plan(), data_version="test")
    independent = portfolio.groupby("loan_type").target_default.agg(["size", "sum", "mean"]).sort_values("mean", ascending=False).iloc[0]
    row = result["rows"][0]
    assert row["loan_type"] == independent.name
    assert row["accounts"] == independent["size"]
    assert row["defaults"] == independent["sum"]
    assert np.isclose(row["default_rate"], independent["mean"])
    assert "loan_accounts.csv" in result["provenance"]["source"]


def test_active_and_question_filters_intersect(portfolio):
    result = execute_plan(portfolio, plan(group_by=[], filters=[{"field": "employment_type", "values": ["Unemployed"]}]), {"region": ["North"], "income_band": ["<30k"]})
    match = portfolio[(portfolio.region == "North") & (portfolio.income_band == "<30k") & (portfolio.employment_type == "Unemployed")]
    assert result["provenance"]["matched_accounts"] == len(match)
    assert result["rows"][0]["defaults"] == match.target_default.sum()
    assert execute_plan(portfolio, plan(), {"region": []})["status"] == "empty"
    assert execute_plan(portfolio, plan(filters=[{"field": "region", "values": ["South"]}]), {"region": ["North"]})["status"] == "empty"


@pytest.mark.parametrize("attack", [plan(metric="__import__('os')"), plan(group_by=["target_default"]), plan(limit=100000), plan(sql="DROP TABLE accounts"), plan(limit="1")])
def test_invalid_plans_never_execute(portfolio, attack):
    with pytest.raises(ValidationError):
        execute_plan(portfolio, attack)


def test_unknown_values_and_duplicate_dimensions(portfolio):
    with pytest.raises(ValueError):
        execute_plan(portfolio, plan(filters=[{"field": "region", "values": ["Atlantis"]}]))
    with pytest.raises(ValueError):
        execute_plan(portfolio, plan(group_by=["region", "region"]))


def test_decline_returns_no_numbers(portfolio):
    result = execute_plan(portfolio, plan(action="decline", decline_reason="future_or_causal"))
    assert result["status"] == "declined"
    assert not result["rows"]
    assert "future" in result["answer"]


def mock_client(query, status="completed", http_status=200, capture=None):
    def handler(request):
        body = json.loads(request.content)
        if capture is not None:
            capture.update(body)
        if http_status != 200:
            return httpx.Response(http_status, json={"error": {"message": "sensitive-provider-error", "type": "invalid_request_error"}})
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 1, "status": status,
            "model": "gpt-4.1-mini", "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(query), "annotations": []}]}],
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        })
    return OpenAI(api_key="fake-unit-test-key", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_real_sdk_wire_contract_and_data_minimization(portfolio):
    sent = {}
    result = ask_portfolio("Which loan type has the highest default rate?", portfolio,
        {"region": ["North"]}, "test", client=mock_client(plan(), capture=sent))
    assert result["status"] == "answered"
    assert sent["store"] is False
    assert sent["text"]["format"]["strict"] is True
    assert sent["text"]["format"]["type"] == "json_schema"
    prompt = json.dumps(sent["input"])
    assert "ACC200001" not in prompt and "CUST100000" not in prompt
    assert "fake-unit-test-key" not in prompt
    assert "601119531" not in prompt


def test_missing_key_no_fake_answer(portfolio, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AssistantUnavailable, match="unavailable"):
        make_plan("Which loan type is highest?", portfolio, {})


@pytest.mark.parametrize("client", [mock_client(plan(), status="incomplete"), mock_client({"malformed": True}), mock_client(plan(), http_status=401)])
def test_provider_failures_are_safe(portfolio, client):
    with pytest.raises(AssistantUnavailable) as exc:
        make_plan("Which loan type?", portfolio, {}, client=client)
    assert "sensitive-provider-error" not in str(exc.value)
    assert "fake-unit-test-key" not in str(exc.value)


@pytest.mark.skipif(os.getenv("RUN_LIVE_LLM_TESTS") != "1", reason="Needs a real server-side API key; opt in with RUN_LIVE_LLM_TESTS=1")
def test_live_llm_end_to_end(portfolio):
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY before opting in"
    answer = ask_portfolio("Which loan type has the highest default rate?", portfolio, {}, "live-test")
    expected = portfolio.groupby("loan_type").target_default.mean().idxmax()
    assert answer["status"] == "answered" and answer["rows"][0]["loan_type"] == expected
    refusal = ask_portfolio("What will the exact default rate be in 2030?", portfolio, {}, "live-test")
    assert refusal["status"] == "declined"
