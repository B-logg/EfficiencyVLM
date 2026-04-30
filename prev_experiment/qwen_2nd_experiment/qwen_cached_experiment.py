import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, LogitsProcessorList
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LOG_DIR = "./runs/qwen_cached_test"
EMBED_DIR = "./qwen_vision_embeddings"
NUM_TEST_SAMPLES = 1010
WARMUP_SAMPLES = 10

writer = SummaryWriter(log_dir=LOG_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"

class CUDATimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    def start(self): self.start_event.record()
    def stop(self): self.end_event.record()
    def get_time(self): return self.start_event.elapsed_time(self.end_event) / 1000.0

class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.first_token_event = torch.cuda.Event(enable_timing=True)
        self.is_first = True
    def __call__(self, input_ids, scores):
        if self.is_first:
            self.first_token_event.record()
            self.is_first = False
        return scores

print("Loading Qwen Model & Tokenizer")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)
image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

sum_times = {k: 0.0 for k in ['db', 'text', 'mlp', 'fusion', 'gen', 'ttft', 'decode', 'latency']}
processed_count = 0

print("Starting Qwen Cached Inference")
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        image_id = data['image_id']
        pt_path = os.path.join(EMBED_DIR, f"embed_{image_id}.pt")
        if not os.path.exists(pt_path): continue
        torch.cuda.reset_peak_memory_stats()

        timer_db = CUDATimer()
        timer_mlp = CUDATimer()
        timer_text = CUDATimer()
        timer_fusion = CUDATimer()
        timer_gen = CUDATimer()

        timer_db.start()
        saved_data = torch.load(pt_path)
        image_embeds = saved_data["embeds"].to(device, dtype=torch.bfloat16)
        grid_thw = saved_data["grid_thw"].to(device)
        timer_db.stop()

        # 2b: MLP Projector (Qwen Merger) -> 1280을 1536으로 투영
        timer_mlp.start()
        
        llm_hidden_size = model.get_input_embeddings().weight.shape[1]
        
        if image_embeds.shape[-1] != llm_hidden_size: # 1280 != 1536 일 경우
            vision_encoder = model.visual if hasattr(model, 'visual') else model.model.visual
            merger = getattr(vision_encoder, 'merger', None)
            if merger is None and hasattr(model, 'model') and hasattr(model.model, 'visual'):
                merger = getattr(model.model.visual, 'merger', None)
            
            if merger is not None:
                image_embeds = merger(image_embeds)
        timer_mlp.stop()

        N_patches = image_embeds.shape[0]

        timer_text.start()
        image_token_str = "<|vision_start|>" + ("<|image_pad|>" * N_patches) + "<|vision_end|>"
        text_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_token_str}Describe this image in detail.<|im_end|>\n<|im_start|>assistant\n"
        text_inputs = processor.tokenizer(text_prompt, return_tensors="pt").to(device)
        timer_text.stop()

        timer_fusion.start()
        inputs_embeds = model.get_input_embeddings()(text_inputs.input_ids)
        image_mask = (text_inputs.input_ids == image_pad_id)
        inputs_embeds[image_mask] = image_embeds
        timer_fusion.stop()

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

        if idx < WARMUP_SAMPLES: continue

        time_db = timer_db.get_time()
        time_text = timer_text.get_time()
        time_mlp = timer_mlp.get_time()
        time_fusion = timer_fusion.get_time()
        time_gen = timer_gen.get_time()

        ttft = timer_gen.start_event.elapsed_time(ttft_processor.first_token_event) / 1000.0
        decode_time = time_gen - ttft
        total_latency = time_db + time_text + time_mlp + time_fusion + time_gen

        generated_tokens = outputs.shape[1]
        tpot = decode_time / max(generated_tokens - 1, 1)
        throughput = generated_tokens / total_latency
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        writer.add_scalar('Pipeline/0_DB_Load', time_db, idx)
        writer.add_scalar('Pipeline/1a_Text', time_text, idx)
        writer.add_scalar('Pipeline/2b_MLP(Merger)', time_mlp, idx)
        writer.add_scalar('Pipeline/3_Fusion', time_fusion, idx)
        writer.add_scalar('LLM_Metrics/TTFT(Prefill)', ttft, idx)
        writer.add_scalar('LLM_Metrics/Decode_Time', decode_time, idx)
        writer.add_scalar('LLM_Metrics/TPOT', tpot, idx)
        writer.add_scalar('LLM_Metrics/Total_Latency', total_latency, idx)
        writer.add_scalar('LLM_Metrics/Throughput', throughput, idx)
        writer.add_scalar('System/VRAM_MB', vram_peak, idx)

        sum_times['db'] += time_db; sum_times['text'] += time_text; sum_times['mlp'] += time_mlp
        sum_times['fusion'] += time_fusion; sum_times['gen'] += time_gen
        sum_times['ttft'] += ttft; sum_times['decode'] += decode_time; sum_times['latency'] += total_latency
        processed_count += 1

if processed_count > 0:
    avg = {k: v / processed_count for k, v in sum_times.items()}
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars1 = ax1.bar(['1a.Text', '1b.Image', '2a.ViT', '2b.MLP(Merger)', '3.Fusion', '4.Gen'], 
                    [avg['text'], 0.0, 0.0, avg['mlp'], avg['fusion'], avg['gen']], color='skyblue')
    ax1.set_title('Qwen Cached Pipeline Average Time')
    ax1.set_ylabel('Seconds')
    for bar in bars1: ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom')
    writer.add_figure('BarCharts/Pipeline_Times', fig1, global_step=0)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars2 = ax2.bar(['TTFT (Prefill)', 'Decode Time', 'Total Latency'], 
                    [avg['ttft'], avg['decode'], avg['latency']], color='coral')
    ax2.set_title('Qwen Cached LLM Inference Metrics')
    ax2.set_ylabel('Seconds')
    for bar in bars2: ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom')
    writer.add_figure('BarCharts/LLM_Metrics', fig2, global_step=0)

    print(f"\nQwen Cached 테스트 완료 (총 {processed_count}장 측정)")
    print(f"평균 TTFT(Prefill): {avg['ttft']:.4f}초 | 평균 Decode: {avg['decode']:.4f}초 | 평균 Total Latency: {avg['latency']:.4f}초")
writer.close()