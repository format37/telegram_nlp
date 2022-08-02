FROM huggingface/transformers-pytorch-gpu
WORKDIR /app
COPY requirements.txt /app
RUN python3 -m pip install -r requirements.txt
COPY loop.py /app
CMD ["python3", "loop.py"]