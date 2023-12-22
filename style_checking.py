import subprocess

subprocess.run(["isort ."], shell=True)
subprocess.run(["mypy ."], shell=True)
subprocess.run(["flake8 ."], shell=True)
subprocess.run(["black ."], shell=True)
