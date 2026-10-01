"""Loading helpers."""

from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st


@contextmanager
def loading(label: str = "Loading operational data…") -> Iterator[None]:
    with st.spinner(label):
        yield
