FROM everymatrix.jfrog.io/emlab-docker-remote-hub/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN mkdir /app
#ADD data.tar /
#COPY vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth

RUN pip3 install \
    tensorboard==2.19.0 \
    matplotlib==3.10.0 \
    torchinfo==1.8.0 \
    einops==0.8.1 \
    huggingface-hub==0.29.1 \
    pytz==2025.1 \
    transformers==4.49.0 \
    typed-argument-parser==1.10.1 \
    pillow==11.1.0 \
    xformers==0.0.29.post3

EXPOSE 6006

VOLUME [ "/app" ]
WORKDIR /app

CMD ["/bin/bash"]