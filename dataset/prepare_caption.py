import os
import random
import torch
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from qwen_omni_utils import process_mm_info

RAW_PATH = <path-to-raw> 

def get_description(video_path, processor, model):
    """Generate description of sounds in video using Qwen2.5-Omni model."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "1. What main sounds are audible in the visible part of the scene?"},
                {"type": "text", "text": "2. What other sounds are audible in the background part of the scene?"}
            ],
        },
    ]
    
    USE_AUDIO_IN_VIDEO = True
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():   
        text_ids = model.generate(
            **inputs, 
            use_audio_in_video=USE_AUDIO_IN_VIDEO, 
            thinker_max_new_tokens=256,
            return_dict_in_generate=True,
            return_audio=False
        )
        text = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1] :], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
    return text

def get_description_all():
    """Generate descriptions for all video files using Qwen2.5-Omni model."""
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B", 
        torch_dtype="auto", 
        device_map="auto"
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    
    data_dir = f"{RAW_PATH}/nfov"
    test_data_list = f"{data_dir}/test"
    train_data_list = f"{data_dir}/train"
    # Find all _av.mp4 files
    video_files = (
        [os.path.join(test_data_list, f) for f in os.listdir(test_data_list) if f.endswith('_av.mp4')] + 
        [os.path.join(train_data_list, f) for f in os.listdir(train_data_list) if f.endswith('_av.mp4')]
    )
    random.shuffle(video_files)
    print(f"Found {len(video_files)} video files.")
    
    for video_file in tqdm(video_files):
        caption_file = video_file.replace('/nfov/', '/caption/').replace('_av.mp4', '_qwen.txt')
        if os.path.exists(caption_file):
            print(f"caption_file exists: {caption_file}")
            continue
        text = get_description(video_file, processor, model)
        print(f"\nvideo_file: {video_file}")
        print(f"caption: {text}")
        os.makedirs(os.path.dirname(caption_file), exist_ok=True)
        with open(caption_file, 'w') as f:
            f.write(text)
       
def get_caption(text_path, tokenizer, model):
    # huggingface-cli download --resume-download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir Qwen/Qwen3-30B-A3B-Instruct-2507
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    with open(text_path, 'r') as f:
        description = f.read().strip()
        
    prompt = description + '\n' + "Summarize the sounds in the scene. Make sure to keep a balance between briefness and clarity. Format the result as: 'Visible sound: <brief description of sound>, Invisible sound: <brief description of background sound>'. If there is no Invisible sound, omit it."

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=16384)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return content

def get_caption_all():
    """Generate captions for all description files using Qwen3 model."""
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    data_dir = f"{RAW_PATH}/caption"
    test_data_list = f"{data_dir}/test"
    train_data_list = f"{data_dir}/train"
    text_files = (
        [os.path.join(test_data_list, f) for f in os.listdir(test_data_list) if f.endswith('_qwen.txt')] + 
        [os.path.join(train_data_list, f) for f in os.listdir(train_data_list) if f.endswith('_qwen.txt')]
    )
    random.shuffle(text_files)
    print(f"Found {len(text_files)} text files.")
    
    for text_file in tqdm(text_files):
        with open(text_file, 'r') as f:
            description = f.read().strip()
            
        caption_file = text_file.replace('_qwen.txt', '_caption.txt')
        if os.path.exists(caption_file):
            print(f"caption_file exists: {caption_file}")
            continue
        text = get_caption(text_file, tokenizer, model)
        print(f"\nProcessing {text_file} ...")
        print(f"description: {description}")
        print(f"caption: {text}")
        os.makedirs(os.path.dirname(caption_file), exist_ok=True)
        with open(caption_file, 'w') as f:
            f.write(text)

def get_tsv():
    data_dir = f"{RAW_PATH}/caption"
    test_data_list = f"{data_dir}/test"
    train_data_list = f"{data_dir}/train"
    test_text_files = [os.path.join(test_data_list, f) for f in os.listdir(test_data_list) if f.endswith('_caption.txt')]
    train_text_files = [os.path.join(train_data_list, f) for f in os.listdir(train_data_list) if f.endswith('_caption.txt')]
    test_text_files.sort()
    train_text_files.sort()
    # random.shuffle(text_files)
    print(f"Found {len(test_text_files)} test_text_files.")
    print(f"Found {len(train_text_files)} train_text_files.")
    test_tsv_file = f"./test.tsv"
    train_tsv_file = f"./train.tsv"
    
    test_values = []
    for text_file in tqdm(test_text_files):
        clip_id = text_file.split('/')[-1].replace('_caption.txt', '')
        with open(text_file, 'r') as f:
            description = f.read().strip()
        if len(description.split('\n')) < 2:
            caption = description.replace('\n', ' ')
        elif len(description.split('\n')) == 2:
            lines = description.split('\n')
            line1 = lines[0].strip()
            line2 = lines[1].strip()
            if line1.endswith('.') or line1.endswith(','):
                caption = line1[:-1] + ', ' + line2
            else:
                caption = line1 + ', ' + line2
        else:
            assert False, print(f"Warning: {text_file} has more than 2 lines.")
        if caption.endswith('.'):
            caption = caption[:-1]
        assert caption.startswith('Visible sound:'), print(f"Warning: {text_file} caption does not start with 'Visible sound:'.")
        test_values.append(f"{clip_id}\t{caption}\n")
    with open(test_tsv_file, 'w') as f:
        f.writelines(test_values)
    
    train_values = []
    for text_file in tqdm(train_text_files):
        clip_id = text_file.split('/')[-1].replace('_caption.txt', '')
        with open(text_file, 'r') as f:
            description = f.read().strip()
        if len(description.split('\n')) < 2:
            caption = description.replace('\n', ' ')
        elif len(description.split('\n')) == 2:
            lines = description.split('\n')
            line1 = lines[0].strip()
            line2 = lines[1].strip()
            if line1.endswith('.') or line1.endswith(','):
                caption = line1[:-1] + ', ' + line2
            else:
                caption = line1 + ', ' + line2
        else:
            assert False, print(f"Warning: {text_file} has more than 2 lines.")
        if caption.endswith('.'):
            caption = caption[:-1]
        assert caption.startswith('Visible sound:'), print(f"Warning: {text_file} caption does not start with 'Visible sound:'.")
        train_values.append(f"{clip_id}\t{caption}\n")
    with open(train_tsv_file, 'w') as f:
        f.writelines(train_values)
    
def format_caption(caption):
    """Format caption by removing meaningless invisible sound descriptions."""
    texts = caption.split(', Invisible sound: ')
    assert len(texts) <= 2, f"Warning: caption has more than 2 parts: {caption}"
    
    if len(texts) == 1:
        return caption
    
    invisible_sound = texts[1].lower()
    filter_words = ['quiet', 'silent', 'silence', 'implied', 'potential', 'possible']
    if any(word in invisible_sound for word in filter_words):
        print(f"Origin caption: {caption}")
        caption = texts[0]
        print(f"Formatted caption: {caption}\n")
    return caption
    
def format_tsv():
    """Format TSV files by cleaning up caption descriptions."""
    data_dir = f"{RAW_PATH}/caption"
    test_tsv_file = f"{data_dir}/test.tsv"
    train_tsv_file = f"{data_dir}/train.tsv"
    
    input_data_dir = "."
    input_test_tsv_file = f"{input_data_dir}/test.tsv"
    input_train_tsv_file = f"{input_data_dir}/train.tsv"
    
    # Process test TSV
    with open(input_test_tsv_file, 'r') as f:
        lines = f.readlines()
    print(f"Read {len(lines)} lines from {input_test_tsv_file}")
    values = []
    for line in lines:
        clip_id, caption = line.strip().split('\t')
        caption = format_caption(caption)
        values.append(f"{clip_id}\t{caption}\n")
    with open(test_tsv_file, 'w') as f:
        f.writelines(values)
    
    # Process train TSV
    with open(input_train_tsv_file, 'r') as f:
        lines = f.readlines()
    print(f"Read {len(lines)} lines from {input_train_tsv_file}")
    values = []
    for line in lines:
        clip_id, caption = line.strip().split('\t')
        caption = format_caption(caption)
        values.append(f"{clip_id}\t{caption}\n")
    with open(train_tsv_file, 'w') as f:
        f.writelines(values)
        
if __name__ == "__main__":
    get_description_all() # get descriptions using Qwen2.5-Omni
    get_caption_all() # get structured captions using Qwen3-Instruct-2507 
    get_tsv() # format to tsv files
    format_tsv() # clean up captions in tsv files