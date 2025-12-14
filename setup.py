# setup.py
from pathlib import Path
from setuptools import setup, find_packages

def parse_requirements():
    req = Path("requirements.txt")
    if req.exists():
        return [
            line.strip()
            for line in req.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return []

setup(
    name="rca-llm",
    version="0.1.0",
    author="Your Name",
    description="RCA LLM utilities (Trainer + utils + HFModelAdapter + RCADataset + Evaluator)",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=parse_requirements(),
    python_requires=">3.9",
)
