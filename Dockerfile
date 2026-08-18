FROM python:3.11.9-slim-bookworm
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir --no-deps .
ENTRYPOINT ["lumitwin-train"]

