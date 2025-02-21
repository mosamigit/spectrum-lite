FROM python:3.9

# sometimes the console won't show print messages,
# using PYTHONUNBUFFERED: 1 can fix this
ENV PYTHONUNBUFFERED=1

WORKDIR /project

COPY . .

RUN python -m pip install -r requirements.txt

# making stdout/err stream unbuffered by passing -u 
# https://docs.python.org/3/library/sys.html#sys.stdout
CMD ["python", "-u", "spectrum_lite_cron.py"]