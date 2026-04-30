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

def get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

print("Loading LLaVA LLM Backbone & Projector")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
image_token_id = tokenizer.convert_tokens_to_ids("<image>")

dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

# GPU 초기화
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

total_pure_inference_time = 0.0
sum_time_db_load = 0.0
sum_time_text_proc = 0.0
sum_time_mlp = 0.0
sum_time_fusion = 0.0
sum_time_generation = 0.0
processed_count = 0

with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        image_id = data['image_id']
        pt_path = os.path.join(EMBED_DIR, f"embed_{image_id}.pt")
        
        if not os.path.exists(pt_path):
            continue
            
        # 0: DB 로드
        t0_load = get_time()
        selected_features = torch.load(pt_path).to(device, dtype=torch.bfloat16)
        time_db_load = get_time() - t0_load
        
        torch.cuda.reset_peak_memory_stats()
        start_time = get_time()

        prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
        
        # 1a: 텍스트 처리
        t0 = get_time()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask 
        time_text_proc = get_time() - t0

        # 1b & 2a (이미지 처리, ViT)
        time_image_proc = 0.0
        time_vit = 0.0

        # 2b: MLP Projector
        t0 = get_time()
        v_proj = model.multi_modal_projector if hasattr(model, 'multi_modal_projector') else model.model.multi_modal_projector
        image_embeds = v_proj(selected_features) 
        time_mlp = get_time() - t0
       
        # 3: Modality Fusion
        t0 = get_time()
        llm_backbone = model
        embed_layer = llm_backbone.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)
        
        image_idx = torch.where(input_ids == image_token_id)[1][0]
        
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
        time_fusion = get_time() - t0
        
        # 4: LLM Generation
        t0 = get_time()
        outputs = llm_backbone.generate(
            inputs_embeds=final_embeds,
            attention_mask=final_mask,
            max_new_tokens=64
        )
        time_generation = get_time() - t0
        
        # 시간 누적 기록
        inference_time = get_time() - start_time
        total_pure_inference_time += inference_time
        sum_time_db_load += time_db_load
        sum_time_text_proc += time_text_proc
        sum_time_mlp += time_mlp
        sum_time_fusion += time_fusion
        sum_time_generation += time_generation
        processed_count += 1
        
        generated_tokens = outputs.shape[1]
        tokens_per_sec = generated_tokens / inference_time
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # 텐서보드 로깅
        writer.add_scalar('Metrics/1_Inference_Time_sec', inference_time, idx)
        writer.add_scalar('Metrics/2_Peak_VRAM_MB', vram_peak, idx)
        writer.add_scalar('Metrics/3_Tokens_Per_Sec', tokens_per_sec, idx)
        writer.add_scalar('Pipeline_Details/1a_Text_Proc_sec', time_text_proc, idx)
        writer.add_scalar('Pipeline_Details/1b_Image_Proc_sec', time_image_proc, idx)
        writer.add_scalar('Pipeline_Details/2a_ViT_sec', time_vit, idx)
        writer.add_scalar('Pipeline_Details/2b_MLP_sec', time_mlp, idx)
        writer.add_scalar('Pipeline_Details/3_Fusion_sec', time_fusion, idx)
        writer.add_scalar('Pipeline_Details/4_Generation_sec', time_generation, idx)
        writer.add_scalar('Pipeline_Details/0_DB_Load_Overhead_sec', time_db_load, idx)



print(f"전체 소요 시간: {total_pure_inference_time:.2f}초")

print("[파이프라인 단계별 평균 소요 시간 (장당)]")
if processed_count > 0:
    print(f"  0. DB 파일 로드 : {sum_time_db_load / processed_count:.5f} 초")
    print(f" 1a. 텍스트 처리  : {sum_time_text_proc / processed_count:.5f} 초")
    print(f" 1b. 이미지 전처리: 0.00000 초 (생략됨)")
    print(f" 2a. ViT 인코딩   : 0.00000 초 (생략됨)")
    print(f" 2b. MLP 투영     : {sum_time_mlp / processed_count:.5f} 초")
    print(f"  3. 모달리티 결합: {sum_time_fusion / processed_count:.5f} 초")
    print(f"  4. 텍스트 생성  : {sum_time_generation / processed_count:.5f} 초")

print("[파이프라인 단계별 평균 소요 시간 (장당)]")
if processed_count > 0:
    print(f"  0. DB 파일 로드 : {sum_time_db_load:.5f} 초")
    print(f" 1a. 텍스트 처리  : {sum_time_text_proc:.5f} 초")
    print(f" 1b. 이미지 전처리: 0.00000 초 (생략됨)")
    print(f" 2a. ViT 인코딩   : 0.00000 초 (생략됨)")
    print(f" 2b. MLP 투영     : {sum_time_mlp:.5f} 초")
    print(f"  3. 모달리티 결합: {sum_time_fusion:.5f} 초")
    print(f"  4. 텍스트 생성  : {sum_time_generation:.5f} 초")


writer.close()