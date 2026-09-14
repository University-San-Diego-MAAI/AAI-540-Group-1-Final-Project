# final-project-usd

## Environment

1. Clone the repo and create the virtualenv:

   ```bash
   git clone <repo-url> && cd final-project-usd
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. Register the Jupyter kernel used by the notebooks:

   ```bash
   .venv/bin/python -m ipykernel install --user --name synpuf-venv
   ```

3. Download the CMS DE-SynPUF OMOP CDM data (public S3 buckets, no credentials needed; ~1 GB total):

   ```bash
   aws s3 sync --no-sign-request s3://synpuf-omop/cmsdesynpuf1k/ data/raw/synpuf1k/
   aws s3 sync --no-sign-request s3://synpuf-omop/cmsdesynpuf100k/ data/raw/synpuf100k/
   ```

4. Open the notebooks:

   ```bash
   .venv/bin/jupyter notebook notebooks/
   ```
