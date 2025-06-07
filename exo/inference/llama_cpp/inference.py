import numpy as np
from pathlib import Path
from typing import Optional

from llama_cpp import Llama

from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader

class LlamaCppInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard_downloader = shard_downloader
        self.shard: Optional[Shard] = None
        self.model: Optional[Llama] = None
        self.tokenizer = None

    async def ensure_shard(self, shard: Shard):
        if self.shard == shard:
            return
        model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
        if Path(model_path).is_dir():
            ggufs = list(Path(model_path).glob('*.gguf'))
            if len(ggufs) == 0:
                raise FileNotFoundError('No .gguf file found in %s' % model_path)
            model_path = ggufs[0]
        self.model = Llama(model_path=str(model_path), logits_all=True)
        self.tokenizer = self.model
        self.shard = shard

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        await self.ensure_shard(shard)
        tokens = self.model.tokenize(prompt.encode('utf-8'), add_bos=False)
        return np.array(tokens, dtype=np.int32)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        await self.ensure_shard(shard)
        return self.model.detokenize(tokens.tolist()).decode('utf-8')

    async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
        if x.ndim == 2:
            logits = x[0]
        else:
            logits = x
        token = int(np.argmax(logits))
        return np.array([token], dtype=np.int32)

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
        await self.ensure_shard(shard)
        tokens = input_data.reshape(-1).astype(int).tolist()
        self.model.eval(tokens)
        logits = self.model._scores[-1:]
        return logits, None

    async def load_checkpoint(self, shard: Shard, path: str):
        pass
