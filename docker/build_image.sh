git clone https://github.com/jupyter/docker-stacks.git
cd docker-stacks/base-notebook
docker build --build-arg ROOT_CONTAINER="nvidia/cuda:11.5.0-runtime-ubuntu20.04" --rm --force-rm -t jupyter-cuda/base-notebook --build-arg NB_USER=takigawa --build-arg NB_UID=1000 --build-arg NB_GID=1000 --build-arg OWNER=jupyter .
cd ../..
docker build -t inokuma/test .
rm -rf docker-stacks
