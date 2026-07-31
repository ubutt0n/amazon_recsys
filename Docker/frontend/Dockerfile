FROM python:3.11

WORKDIR /app

RUN pip install streamlit

COPY src/frontend /app/

CMD streamlit run --server.port 8080 ui.py