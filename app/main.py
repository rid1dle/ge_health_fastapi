import subprocess

if __name__ == "__main__":
    subprocess.run(
        "gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.api:app --bind 0.0.0.0:4500 --timeout 0".split()
    )
