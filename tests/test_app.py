"""Tests for the Streamlit app.

The app is exercised with Streamlit's own testing harness, which runs the
script the way the browser does. Nothing here touches the network: the
optimization only runs when the button is pressed, and the tests that press
it use an empty universe on purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture()
def app() -> AppTest:
    """A freshly run instance of the app."""
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def test_the_app_renders_without_error(app: AppTest) -> None:
    assert not app.exception


def test_the_app_shows_its_title(app: AppTest) -> None:
    assert app.title[0].value == "Mean-Variance Portfolio Optimizer"


def test_the_app_has_a_run_button(app: AppTest) -> None:
    assert any(button.label == "Run optimization" for button in app.button)


def test_the_app_asks_for_a_run_before_showing_results(app: AppTest) -> None:
    messages = [info.value for info in app.info]

    assert any("Run optimization" in message for message in messages)


def test_the_app_has_the_five_result_tabs(app: AppTest) -> None:
    labels = [tab.label for tab in app.tabs]

    assert labels == ["Portfolio", "Risk", "Positions", "Validation", "History"]


def test_the_sidebar_collects_the_universe_and_the_objective(app: AppTest) -> None:
    sidebar = app.sidebar

    assert any("Tickers" in element.label for element in sidebar.text_area)
    assert any("Risk-free rate" in element.label for element in sidebar.number_input)
    assert any("Expected returns" in element.label for element in sidebar.selectbox)


def test_pressing_run_without_tickers_reports_an_error(app: AppTest) -> None:
    app.text_area[0].set_value("")
    app.button[0].click().run()

    assert not app.exception
    assert any("at least one ticker" in error.value for error in app.error)
