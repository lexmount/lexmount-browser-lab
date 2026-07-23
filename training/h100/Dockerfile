# CUDA/H100 training image for the LexBrowser WebVoyager GRPO recipe.
#
# This mirrors the internal Ascend training image, which built on
# quay.io/ascend/vllm-ascend:nightly-releases-v0.18.0 (vLLM 0.18.0 +
# vllm-ascend 0.18.1.dev41). On CUDA the matching base is the official
# vLLM 0.18.0 image; vllm-ascend simply drops out.
#
# Build context is the training/h100/ subtree:
#   docker build -f training/h100/Dockerfile -t lexbrowser-verl-h100:local training/h100
#
# Version provenance (from the validated 2026-07-21 Ascend run image):
#   verl          0.9.0.dev0  -> pinned by git commit below (not on PyPI)
#   vllm          0.18.0      -> base image (CUDA build ships torch 2.10.0;
#                                the Ascend image ran torch 2.9.0 because
#                                torch_npu was pinned there - see PORTING.md)
#   transformers  5.3.0       -> the validated image ran 5.3.0.dev0; 5.3.0 is
#                                the matching PyPI release
FROM vllm/vllm-openai:v0.18.0

# Same verl commit as the validated Ascend image (verl version 0.9.0.dev).
ARG VERL_COMMIT=30119a253087bff86c12d329d2d8dd43c589705f

RUN git clone https://github.com/verl-project/verl.git /verl && \
    git -C /verl checkout "$VERL_COMMIT" && \
    pip install --no-cache-dir -r /verl/requirements.txt && \
    pip install --no-cache-dir --no-deps -e /verl && \
    rm -rf /root/.cache/pip

# Match the validated run's transformers line (Ascend image ran 5.3.0.dev0).
# vllm 0.18.0 declares transformers<5; the validated stack already ran with
# transformers 5.x over vllm 0.18, so we intentionally follow the validated
# stack rather than vllm's declared bound. Verify on first H100 smoke run.
RUN pip install --no-cache-dir "transformers==5.3.0"

WORKDIR /workspace/lexbrowser-h100
COPY runtime/lexbrowser_webvoyager /tmp/lexbrowser_webvoyager
RUN pip install --no-cache-dir /tmp/lexbrowser_webvoyager && rm -rf /tmp/lexbrowser_webvoyager

# The vllm-openai base image's entrypoint starts an API server; this image is
# driven by Ray/verl instead.
ENTRYPOINT []
CMD ["bash"]
