import os
import torch
import matplotlib
matplotlib.use('Agg') # GUI 없이 그래프 생성
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, LogitsProcessorList
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LOG_DIR = "./runs/qwen_e2e_test"
NUM_TEST_SAMPLES = 1010 # 10장 웜업 + 1000장 테스트
WARMUP_SAMPLES = 10

writer = SummaryWriter(log_dir=LOG_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"

class CUDATimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    def start(self):
        self.start_event.record()
    def stop(self):
        self.end_event.record()
    def get_time(self):
        return self.start_event.elapsed_time(self.end_event) / 1000.0

class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.first_token_event = torch.cuda.Event(enable_timing=True)
        self.is_first = True
    def __call__(self, input_ids, scores):
        if self.is_first:
            self.first_token_event.record()
            self.is_first = False
        return scores

print("Loading Qwen Model & Processor")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

sum_times = {k: 0.0 for k in ['text', 'img', 'vit', 'mlp', 'fusion', 'gen', 'ttft', 'decode', 'latency']}
processed_count = 0

print("Starting Qwen E2E Inference...")
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        image = data['image'].convert("RGB") if data['image'].mode != "RGB" else data['image']
        torch.cuda.reset_peak_memory_stats()
        
        timer_text = CUDATimer()
        timer_img = CUDATimer()
        timer_vit = CUDATimer()
        timer_mlp = CUDATimer()
        timer_fusion = CUDATimer()
        timer_gen = CUDATimer()
        
        # 1b: Image Proc
        timer_img.start()
        image_inputs = processor.image_processor(images=image, return_tensors="pt").to(device)
        pixel_values = image_inputs.pixel_values.to(dtype=torch.bfloat16)
        grid_thw = image_inputs.image_grid_thw
        timer_img.stop()

        # 2a: ViT
        timer_vit.start()
        vision_encoder = model.visual if hasattr(model, 'visual') else model.model.visual
        vision_outputs = vision_encoder(pixel_values, grid_thw=grid_thw)
        
        # 박스에서 텐서 추출
        if hasattr(vision_outputs, 'last_hidden_state'):
            image_embeds = vision_outputs.last_hidden_state
        elif isinstance(vision_outputs, tuple):
            image_embeds = vision_outputs[0]
        else:
            image_embeds = vision_outputs
        timer_vit.stop()

        # 2b: MLP Projector: 1280 => 1536으로 투영
        timer_mlp.start()

        llm_hidden_size = model.get_input_embeddings().weight.shape[1]

        if image_embeds.shape[-1] != llm_hidden_size: # 1280 != 1536 일 경우
            merger = getattr(vision_encoder, 'merger', None)
            if merger is None and hasattr(model, 'model') and hasattr(model.model, 'visual'):
                merger = getattr(model.model.visual, 'merger', None)
            
            if merger is not None:
                image_embeds = merger(image_embeds)
        timer_mlp.stop()
        
        N_patches = image_embeds.shape[0]

        # 1a: Text Proc
        timer_text.start()
        image_token_str = "<|vision_start|>" + ("<|image_pad|>" * N_patches) + "<|vision_end|>"
        text_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_token_str}Describe this image in detail.<|im_end|>\n<|im_start|>assistant\n"
        text_inputs = processor.tokenizer(text_prompt, return_tensors="pt").to(device)
        timer_text.stop()

        # 3: Fusion
        timer_fusion.start()
        inputs_embeds = model.get_input_embeddings()(text_inputs.input_ids)
        image_mask = (text_inputs.input_ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        inputs_embeds[image_mask] = image_embeds
        timer_fusion.stop()

        # 4: LLM Generation (with TTFT)
        ttft_processor = TTFTLogitsProcessor()
        processors = LogitsProcessorList([ttft_processor])
        
        timer_gen.start()
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=text_inputs.attention_mask,
            image_grid_thw=grid_thw,
            max_new_tokens=64,
            logits_processor=processors
        )
        timer_gen.stop()
        
        torch.cuda.synchronize()
        
        if idx < WARMUP_SAMPLES:
            continue
            
        time_text = timer_text.get_time()
        time_img = timer_img.get_time()
        time_vit = timer_vit.get_time()
        time_mlp = timer_mlp.get_time()
        time_fusion = timer_fusion.get_time()
        time_gen = timer_gen.get_time()
        
        ttft = timer_gen.start_event.elapsed_time(ttft_processor.first_token_event) / 1000.0
        decode_time = time_gen - ttft
        total_latency = time_text + time_img + time_vit + time_mlp + time_fusion + time_gen
        
        generated_tokens = outputs.shape[1]
        tpot = decode_time / max(generated_tokens - 1, 1)
        throughput = generated_tokens / total_latency
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # 텐서보드 기록
        writer.add_scalar('Pipeline/1a_Text', time_text, idx)
        writer.add_scalar('Pipeline/1b_Image', time_img, idx)
        writer.add_scalar('Pipeline/2a_ViT', time_vit, idx)
        writer.add_scalar('Pipeline/2b_MLP(Merger)', time_mlp, idx)
        writer.add_scalar('Pipeline/3_Fusion', time_fusion, idx)
        writer.add_scalar('LLM_Metrics/TTFT(Prefill)', ttft, idx)
        writer.add_scalar('LLM_Metrics/Decode_Time', decode_time, idx)
        writer.add_scalar('LLM_Metrics/TPOT', tpot, idx)
        writer.add_scalar('LLM_Metrics/Total_Latency', total_latency, idx)
        writer.add_scalar('LLM_Metrics/Throughput', throughput, idx)
        writer.add_scalar('System/VRAM_MB', vram_peak, idx)

        sum_times['text'] += time_text; sum_times['img'] += time_img
        sum_times['vit'] += time_vit; sum_times['mlp'] += time_mlp
        sum_times['fusion'] += time_fusion; sum_times['gen'] += time_gen
        sum_times['ttft'] += ttft; sum_times['decode'] += decode_time; sum_times['latency'] += total_latency
        processed_count += 1

if processed_count > 0:
    avg = {k: v / processed_count for k, v in sum_times.items()}
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars1 = ax1.bar(['1a.Text', '1b.Image', '2a.ViT', '2b.MLP(Merger)', '3.Fusion', '4.Gen'], 
                    [avg['text'], avg['img'], avg['vit'], avg['mlp'], avg['fusion'], avg['gen']], color='skyblue')
    ax1.set_title('Qwen E2E Pipeline Average Time')
    ax1.set_ylabel('Seconds')
    for bar in bars1: ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom')
    writer.add_figure('BarCharts/Pipeline_Times', fig1, global_step=0)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars2 = ax2.bar(['TTFT (Prefill)', 'Decode Time', 'Total Latency'], 
                    [avg['ttft'], avg['decode'], avg['latency']], color='coral')
    ax2.set_title('Qwen E2E LLM Inference Metrics')
    ax2.set_ylabel('Seconds')
    for bar in bars2: ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom')
    writer.add_figure('BarCharts/LLM_Metrics', fig2, global_step=0)

    print(f"\nQwen E2E 테스트 완료 (총 {processed_count}장 측정, WARMUP 10장 제외)")
    print(f"평균 TTFT(Prefill): {avg['ttft']:.4f}초 | 평균 Decode: {avg['decode']:.4f}초 | 평균 Total Latency: {avg['latency']:.4f}초")

writer.close()