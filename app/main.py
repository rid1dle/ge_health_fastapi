import subprocess

if __name__ == "__main__":
    # subprocess.run("docker-compose up -d".split(), shell=True)
    subprocess.run(
        "gunicorn -w 1 -k uvicorn.workers.UvicornWorker app.api:app --bind 0.0.0.0:4500 --timeout 0".split()
    )
    # subprocess.run(
    #     "uvicorn app.api:app --host 0.0.0.0 --port 4500 --ws-ping-timeout 200000".split()
    # )
