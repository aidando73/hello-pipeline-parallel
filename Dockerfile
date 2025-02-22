# Dockerfile

FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/pytorch/PiPPy.git \
    && cd PiPPy \
    && python setup.py install \
    && cd ..

COPY . .

ENTRYPOINT ["python", "example.py"]