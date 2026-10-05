"""
Make the app read its key from Streamlit secrets as well as the environment.

    python patch_secrets.py

Locally the key comes from .env or an exported variable. On Streamlit Cloud
there is no environment to export into, so it comes from st.secrets, which
you paste into App settings, Secrets.
"""
from pathlib import Path

LOADER = '''
def _secret(name, default=None):
    """Streamlit Cloud supplies secrets through st.secrets. Locally the
    environment or a .env file supplies them. Try both."""
    v = os.getenv(name)
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default

'''

p = Path("app/generate.py")
s = p.read_text()
if "_secret(" in s:
    print("  already patched")
else:
    s = s.replace("import os\n", "import os\n" + LOADER, 1)
    s = s.replace('os.environ["GROQ_API_KEY"]', '_secret("GROQ_API_KEY")')
    s = s.replace('os.getenv("MODEL", "openai/gpt-oss-120b")',
                  '_secret("MODEL", "llama-3.3-70b-versatile")')
    s = s.replace('if not os.getenv("GROQ_API_KEY"):',
                  'if not _secret("GROQ_API_KEY"):')
    p.write_text(s)
    print("  app/generate.py now reads st.secrets and the environment")
