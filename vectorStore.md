```sh
docker pull qdrant/qdrant
```

```sh
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

```sh
docker model run ai/qwen2.5:3B-Q4_K_M
```