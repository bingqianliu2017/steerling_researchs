from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download(
    model_id="guidelabs/steerling-8b",  # 核心要替换成 ModelScope 上实际存在的模型 ID
    cache_dir="/root/models/steerling-8b"
)
print("模型下载到:", model_dir)


from steerling import SteerlingGenerator, GenerationConfig

# 这里改成本地下载目录
generator = SteerlingGenerator.from_pretrained( "/root/models/steerling-8b/guidelabs/steerling-8b")

text = generator.generate(
    "what is machine learning?",
    GenerationConfig(max_new_tokens=100, seed=42),
)
print(text)