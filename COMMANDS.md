```bash
source ~/miniconda3/bin/activate
conda create --prefix ./env
source ~/miniconda3/bin/activate ./env

sudo docker build -t example .
sudo docker network create rpc
sudo docker run -e RANK=0 -e WORLD=2 \
  -e MASTER_ADDR=head -e MASTER_PORT=3000 \
  --net rpc --name head --rm -it example

sudo docker run -e RANK=1 -e WORLD=2 \
  -e MASTER_ADDR=head -e MASTER_PORT=3000 \
  --net rpc --name worker --rm -it example

pip install torch torchvision
```