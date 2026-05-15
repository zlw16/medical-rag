"""Wrapper to start the server and log all output"""
import sys
import os

# Log all output to a file
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_wrapper.log")
sys.stdout = open(log_file, "w", encoding="utf-8", buffering=1)
sys.stderr = sys.stdout

print("="*60)
print("Starting Medical RAG Server Wrapper")
print("="*60)

# Now import and run start_app
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import start_app
