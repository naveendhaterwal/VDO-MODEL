import os
import torch
import gc
import logging
from monitoring.memory_watchdog import MemoryWatchdog

logger = logging.getLogger(__name__)
MODEL_PATH = os.path.join(os.environ.get("MODEL_DIR", "/app/models"), "Qwen2.5-7B-Instruct")


class LocalChatProvider:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def __enter__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading {self.model_path} into VRAM...")
        MemoryWatchdog.assert_vram_available(required_gb=14.0)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16

        logger.info(f"GPU compute capability: {cap}, using dtype: {dtype}")

        # --- Strategy 1: 4-bit quantization via bitsandbytes (lowest VRAM) ---
        # Falls back if the node GPU arch doesn't have bitsandbytes kernels
        if torch.cuda.is_available():
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    device_map="auto",
                    quantization_config=bnb_config,
                )
                logger.info("Model loaded successfully with 4-bit quantization (BnB).")
                return self
            except Exception as e:
                logger.warning(
                    f"BitsAndBytes 4-bit quantization failed on this GPU (arch={cap}): {e}\n"
                    f"Falling back to float16 without quantization..."
                )
                # Clear any partial state before retrying
                if self.model is not None:
                    del self.model
                    self.model = None
                gc.collect()
                torch.cuda.empty_cache()

        # --- Strategy 2: Standard float16 load (universal GPU compatibility) ---
        # Uses more VRAM (~14GB vs ~7GB for 4-bit) but runs on ALL CUDA GPUs
        logger.info("Loading model in float16 without quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        logger.info("Model loaded successfully in float16 (no quantization).")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Unloading Chat Model and freeing VRAM...")
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        MemoryWatchdog.enforce_cleanup()

    def generate(self, prompt: str, system_prompt: str = "You are a professional cinematic screenwriter.") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        logger.info("Generating response...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1500,
                temperature=0.7,
                do_sample=True,
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
