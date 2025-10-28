# stop qdrant containers (keep as before)
docker ps -aq --filter ancestor=qdrant/qdrant | xargs -r docker rm -f

# stop all containers whose image starts with "ai/"
docker ps -a --format '{{.ID}} {{.Image}}' \
  | awk '$2 ~ /^ai\// {print $1}' \
  | xargs -r docker rm -f