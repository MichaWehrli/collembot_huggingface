FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 wget p7zip-full \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 collembot
WORKDIR /app
RUN chown collembot:collembot /app
USER collembot

ENV PATH="/home/collembot/.local/bin:$PATH"

# Install CUDA-enabled PyTorch (cu121 works on L4 / T4 / A10G / H100)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir \
        "numpy<2" \
        ultralytics \
        opencv-python-headless \
        Pillow \
        scipy \
        scikit-learn \
        "shapely>=2.0" \
        PyYAML \
        tqdm \
        openpyxl \
        matplotlib \
        gradio>=4.0 \
        spaces \
        requests \
        pandas

RUN git clone https://github.com/waldstrom/collembot.git /tmp/collembot \
 && cp -r /tmp/collembot/scripts  /app/scripts  \
 && cp -r /tmp/collembot/configs  /app/configs  \
 && cp -r /tmp/collembot/data     /app/data     \
 && rm -rf /tmp/collembot

COPY --chown=collembot:collembot . .
# Ensure the Space's gradio_app.py overrides the cloned one
RUN if [ -f /app/gradio_app.py ]; then cp /app/gradio_app.py /app/scripts/gradio_app.py; fi

EXPOSE 7860
ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

CMD python app.py