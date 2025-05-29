
docker run --rm -it --gpus all --network host \
           --ulimit memlock=-1 --ulimit stack=67108864 \
           -v "$(pwd)":/app equiface:latest bash

