FROM python:3.10-slim
COPY . /main
WORKDIR /main
RUN pip config list
RUN cat /main/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000"]