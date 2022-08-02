###
# This shell script connects to container via ssh
###

# get id of running container with image name "jperldev/dain"
container_id=$(docker ps | grep docker_transformers | awk '{print $1}')
echo "container id: $container_id"
# connect to container
docker exec -it $container_id /bin/bash
