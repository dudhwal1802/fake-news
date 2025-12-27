"""Streamlit Community Cloud entrypoint.

Streamlit Cloud often expects a file named `streamlit_app.py` by default.
Our main app UI lives in `app.py`, so this file simply delegates to it.
"""

from app import main


main()
