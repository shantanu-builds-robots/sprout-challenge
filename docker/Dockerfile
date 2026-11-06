# The reference environment: the grader runs every submission in this image, and entrants may use it to check
# theirs behaves the same. Python 3.11, MuJoCo, numpy, scipy and onnxruntime (for exported neural networks).
#   docker build -f docker/Dockerfile -t sprout-challenge .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install "numpy<3" scipy onnxruntime
WORKDIR /opt/sprout-challenge
COPY pyproject.toml README.md LICENSE ./
COPY sprout_challenge ./sprout_challenge
RUN pip install . && python -c "from sprout_challenge import Sim, Track; Sim(Track.load('flat')).reset(0)"
WORKDIR /work
CMD ["python", "-m", "sprout_challenge.grade", "--help"]
