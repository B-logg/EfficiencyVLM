import os
import time
import torch
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LOG_DIR = "./runs/llava_cached_test"
EMBED_DIR = "./llava_vision_embeddings" 
NUM_TEST_SAMPLES = 500

writer = SummaryWriter(log_dir=LOG_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading LLaVA LLM Backbone & Projector...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
image_token_id = tokenizer.convert_tokens_to_ids("<image>")

dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

print("Starting LLaVA Cached Inference...")
total_pure_inference_time = 0.0

with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        image_id = data['image_id']
        pt_path = os.path.join(EMBED_DIR, f"embed_{image_id}.pt")
        
        if not os.path.exists(pt_path):
            continue
            
        # 1. 캐싱된 1024차원 순수 시각 토큰 로드
        selected_features = torch.load(pt_path).to(device, dtype=torch.bfloat16)
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        # 추론 시점에 Projector 통과시키기
        v_proj = model.multi_modal_projector if hasattr(model, 'multi_modal_projector') else model.model.multi_modal_projector
        image_embeds = v_proj(selected_features) 


        prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        if hasattr(model, 'language_model'):
            llm_backbone = model.language_model
        elif hasattr(model, 'text_model'):
            llm_backbone = model.text_model
        elif hasattr(model, 'model') and hasattr(model.model, 'text_model'):
            llm_backbone = model.model.text_model
        else:
            llm_backbone = model
            
        if hasattr(llm_backbone, 'get_input_embeddings'):
            embed_layer = llm_backbone.get_input_embeddings()
        else:
            embed_layer = model.get_input_embeddings()

        # 임베딩 레이어로 텍스트를 벡터화
        inputs_embeds = embed_layer(input_ids)
        
        image_idx = torch.where(input_ids == image_token_id)[1][0]
        
        # 모달리티 결합
        final_embeds = torch.cat([
            inputs_embeds[:, :image_idx, :],
            image_embeds,
            inputs_embeds[:, image_idx+1:, :]
        ], dim=1)
        
        image_mask = torch.ones((1, image_embeds.shape[1]), dtype=attention_mask.dtype, device=device)
        final_mask = torch.cat([
            attention_mask[:, :image_idx],
            image_mask,
            attention_mask[:, image_idx+1:]
        ], dim=1)
        
        outputs = llm_backbone.generate(
            inputs_embeds=final_embeds,
            attention_mask=final_mask,
            max_new_tokens=64
        )
        
        inference_time = time.time() - start_time
        total_pure_inference_time += inference_time
        
        generated_tokens = outputs.shape[1]
        tokens_per_sec = generated_tokens / inference_time
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        writer.add_scalar('Metrics/1_Inference_Time_sec', inference_time, idx)
        writer.add_scalar('Metrics/2_Peak_VRAM_MB', vram_peak, idx)
        writer.add_scalar('Metrics/3_Tokens_Per_Sec', tokens_per_sec, idx)

print(f"총 소요 시간: {total_pure_inference_time:.2f}초")
writer.close()