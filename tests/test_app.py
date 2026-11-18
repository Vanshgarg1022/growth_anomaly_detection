import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(PROJECT_ROOT, "app.py")


def page_text(at: AppTest) -> str:
    """Helper to join all text element values into one searchable string."""
    texts = []
    for el in at.markdown:
        texts.append(el.value)
    for el in at.caption:
        texts.append(el.value)
    for el in at.info:
        texts.append(el.value)
    for el in at.warning:
        texts.append(el.value)
    for el in at.error:
        texts.append(el.value)
    return "\n".join(texts)


def test_validation_errors_stop_the_calculation():
    at = AppTest.from_file(APP_PATH, default_timeout=10)
    at.run()

    # Clear patient code input to trigger Patient Code error
    at.text_input[0].input("").run()

    # Click form submit button
    at.button[0].click().run()

    txt = page_text(at)
    assert "Patient Code is required." in txt
    assert "Weight is required and must be > 0 kg." in txt


def test_app_loads_without_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=10)
    at.run()
    assert not at.exception
