FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       make gcc-riscv64-linux-gnu libc6-dev-riscv64-cross \
    && rm -rf /var/lib/apt/lists/*
