FROM python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libegl-mesa0 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY competition/requirements.txt /app/competition/requirements.txt
RUN pip install --no-cache-dir -r competition/requirements.txt
COPY competition /app/competition
COPY roboquest /app/roboquest
COPY models/go2 /app/models/go2
RUN useradd --uid 10001 --create-home rq
USER rq
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1 MUJOCO_GL=egl
ENTRYPOINT ["python", "-m", "competition.run_match"]
