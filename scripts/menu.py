"""Unified master menu entry point.

Usage:
    python scripts/menu.py
"""
from cola_coder.session_log import start_session_log
from cola_coder.features.master_menu import run_master_menu

if __name__ == "__main__":
    session = start_session_log()
    print(f"  Session log: {session.log_path}")
    try:
        run_master_menu()
    finally:
        session.close()
