"""Root-level shim — delegates to the package CLI entry point."""

from ai_skill.__main__ import app

if __name__ == "__main__":
    app()
