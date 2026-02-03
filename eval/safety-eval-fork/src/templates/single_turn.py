from fastchat.conversation import get_conv_template
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer

ALPACA_PROMPT = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
}
VICUNA_1_0_PROMPT = {
    "description": "Template used by Vicuna 1.0 and stable vicuna.",
    "prompt": "### Human: {instruction}\n### Assistant:",
}

VICUNA_PROMPT = {
    "description": "Template used by Vicuna.",
    "prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {instruction} ASSISTANT:",
}

OASST_PROMPT = {
    "description": "Template used by Open Assistant",
    "prompt": "<|prompter|>{instruction}<|endoftext|><|assistant|>"
}
OASST_PROMPT_v1_1 = {
    "description": "Template used by newer Open Assistant models",
    "prompt": "<|prompter|>{instruction}</s><|assistant|>"
}

LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
LLAMA2_CHAT_PROMPT = {
    "description": "Template used by Llama2 Chat",
    # "prompt": "[INST] {instruction} [/INST] "
    "prompt": "[INST] <<SYS>>\n" + LLAMA2_DEFAULT_SYSTEM_PROMPT + "\n<</SYS>>\n\n{instruction} [/INST] "
}

LLAMA2_CHAT_PROMPT_NO_SYS = {
    "description": "Template used by Llama2 Chat",
    "prompt": "[INST] {instruction} [/INST] "
}

INTERNLM_PROMPT = {  # https://github.com/InternLM/InternLM/blob/main/tools/alpaca_tokenizer.py
    "description": "Template used by INTERNLM-chat",
    "prompt": "<|User|>:{instruction}<eoh><|Bot|>:"
}

KOALA_PROMPT = {  # https://github.com/young-geng/EasyLM/blob/main/docs/koala.md#koala-chatbot-prompts
    "description": "Template used by EasyLM/Koala",
    "prompt": "BEGINNING OF CONVERSATION: USER: {instruction} GPT:"
}

# Get from Rule-Following: cite
FALCON_PROMPT = {  # https://huggingface.co/tiiuae/falcon-40b-instruct/discussions/1#6475a107e9b57ce0caa131cd
    "description": "Template used by Falcon Instruct",
    "prompt": "User: {instruction}\nAssistant:",
}

MPT_PROMPT = {  # https://huggingface.co/TheBloke/mpt-30B-chat-GGML
    "description": "Template used by MPT",
    "prompt": '''<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|><|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n''',
}

DOLLY_PROMPT = {
    "description": "Template used by Dolly",
    "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
}

OPENAI_CHATML_PROMPT = {
    "description": "Template used by OpenAI chatml",  # https://github.com/openai/openai-python/blob/main/chatml.md
    "prompt": '''<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
'''
}

LLAMA2_70B_OASST_CHATML_PROMPT = {
    "description": "Template used by OpenAI chatml",  # https://github.com/openai/openai-python/blob/main/chatml.md
    "prompt": '''<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''
}

FALCON_INSTRUCT_PROMPT = {  # https://huggingface.co/tiiuae/falcon-40b-instruct/discussions/1#6475a107e9b57ce0caa131cd
    "description": "Template used by Falcon Instruct",
    "prompt": "User: {instruction}\nAssistant:",
}

FALCON_CHAT_PROMPT = {  # https://huggingface.co/blog/falcon-180b#prompt-format
    "description": "Template used by Falcon Chat",
    "prompt": "User: {instruction}\nFalcon:",
}

ORCA_2_PROMPT = {
    "description": "Template used by microsoft/Orca-2-13b",
    "prompt": "<|im_start|>system\nYou are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant"
}

MISTRAL_PROMPT = {
    "description": "Template used by Mistral Instruct",
    "prompt": "[INST] {instruction} [/INST]"
}

BAICHUAN_CHAT_PROMPT = {
    "description": "Template used by Baichuan2-chat",
    "prompt": "<reserved_106>{instruction}<reserved_107>"
}

QWEN_CHAT_PROMPT = {
    "description": "Template used by Qwen-chat models",
    "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
}

ZEPHYR_PROMPT = {
    "description": "",
    "prompt": "<|user|>\n{instruction}</s>\n<|assistant|>\n"
}

MIXTRAL_PROMPT = {
    "description": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "prompt": "[INST] {instruction} [/INST]"
}

TULU2_PROMPT = {
    "description": "Template used by Tulu2 SFT and DPO",
    "prompt": "<|user|>\n{instruction}\n<|assistant|>\n",
}

OLMO_CHAT_PROMPT = {
    "description": "Template used by OLMo Chat models SFT and DPO",
    "prompt": "<|endoftext|><|user|>\n{instruction}\n<|assistant|>\n",
}
LLAMA3_CHAT_PROMPT = {
    "description": "Template used by Llama3 instruction-tuned models",
    "prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}

cot_format = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.\n\nUser: {instruction}"""

LLAMA3_COT_PROMPT = {
    "description": "Template used by Llama3 instruction-tuned models",
    "prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + cot_format + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\n<think> "
}

# cot_defender_system = "You are helpful and harmless assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
# cot_defender_format = """**Your goal is to react correctly to the nature of the request: comply with benign requests while defending against jailbreak attacks**. To clarify the definition, "jailbreaking" a language model means that users strategically revise a simple harmful prompt into a more complicated version so that the revised prompt can bypass the safeguard of the language model and drive the language to produce harmful responses.

# **Response Format**:
# You first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You could only use <think> </think> and <answer> </answer> tags once. Please follow the format strictly, otherwise you will lose rewards.

# User: {instruction}"""

# LLAMA3_COT_DEFENDER_PROMPT = {
#     "description": "Template used by Llama3 instruction-tuned models",
#     "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + cot_defender_system  + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + cot_defender_format + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + "Let me think step by step.\n<think>"
# }
deepseek_r1_distill_llama3_system = """Respond in the following format:
<think>
You should reason between these tags.
</think>

Answer goes here...

Always use <think> </think> tags even if they are not necessary.
"""
DEEPSEEK_R1_DISTILL_LLAMA3_PROMPT = {
    "description": "Template used by DeepSeek R1 Distill Llama",
    "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{deepseek_r1_distill_llama3_system}<|eot_id|>\n\n" + "<|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|>\n\n" + "<|start_header_id|>assistant<|end_header_id|>\n<think>"
}
#You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
#You are a helpful AI assistant. The user will ask you a question. Please first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.

# Qwen-Game defender template
# GAME_DEFENDER_SYSTEM = """You are a helpful AI assistant.The user will ask you a question. Please first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.
# """.strip()

GAME_DEFENDER_SYSTEM = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
""".strip()

SELF_REDTEAM_USER_SYSTEM = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within < think> </think> and <answer> </answer> tags, respectively.
User: {instruction}
Let‘s think step by step. <think>
""".strip()

DEFENDER_FIRST_USER_PROMPT = """"
{instruction}
""".strip()
#A conversation between User and Assistant.The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
#You are a helpful AI assistant. The user will ask you a question. Please first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.



########## CHAT TEMPLATE ###########

def get_template(model_name_or_path=None, chat_template=None, fschat_template=None, system_message=None,
                 return_fschat_conv=False, **kwargs):
    # ==== First check for fschat template ====
    if fschat_template or return_fschat_conv:
        fschat_conv = _get_fschat_conv(model_name_or_path, fschat_template, system_message)
        if return_fschat_conv:
            print("Found FastChat conv template for", model_name_or_path)
            print(fschat_conv.dict())
            return fschat_conv
        else:
            fschat_conv.append_message(fschat_conv.roles[0], "{instruction}")
            fschat_conv.append_message(fschat_conv.roles[1], None)
            TEMPLATE = {"description": f"fschat template {fschat_conv.name}", "prompt": fschat_conv.get_prompt()}
    # ===== Check for some older chat model templates ====
    elif chat_template == "wizard":
        TEMPLATE = VICUNA_PROMPT
    elif chat_template == "vicuna":
        TEMPLATE = VICUNA_PROMPT
    elif chat_template == "oasst":
        TEMPLATE = OASST_PROMPT
    elif chat_template == "oasst_v1_1":
        TEMPLATE = OASST_PROMPT_v1_1
    elif chat_template == "llama-2":
        TEMPLATE = LLAMA2_CHAT_PROMPT
    elif chat_template == "llama-2_no_sys":
        TEMPLATE = LLAMA2_CHAT_PROMPT_NO_SYS
    elif chat_template == "falcon_instruct":  # falcon 7b / 40b instruct
        TEMPLATE = FALCON_INSTRUCT_PROMPT
    elif chat_template == "falcon_chat":  # falcon 180B_chat
        TEMPLATE = FALCON_CHAT_PROMPT
    elif chat_template == "mpt":
        TEMPLATE = MPT_PROMPT
    elif chat_template == "koala":
        TEMPLATE = KOALA_PROMPT
    elif chat_template == "dolly":
        TEMPLATE = DOLLY_PROMPT
    elif chat_template == "internlm":
        TEMPLATE = INTERNLM_PROMPT
    elif chat_template == "mistral" or chat_template == "mixtral":
        TEMPLATE = MISTRAL_PROMPT
    elif chat_template == "orca-2":
        TEMPLATE = ORCA_2_PROMPT
    elif chat_template == "baichuan2":
        TEMPLATE = BAICHUAN_CHAT_PROMPT
    elif chat_template == "qwen":
        TEMPLATE = QWEN_CHAT_PROMPT
    elif chat_template == "zephyr":
        TEMPLATE = ZEPHYR_PROMPT
    elif chat_template == "tulu2":
        TEMPLATE = TULU2_PROMPT
    elif chat_template == "olmo":
        TEMPLATE = OLMO_CHAT_PROMPT
    elif chat_template == "llama3" or chat_template == 'llama-3':
        TEMPLATE = LLAMA3_CHAT_PROMPT
    elif chat_template == "phi3" or chat_template == 'phi-3':
        TEMPLATE = PHI3_CHAT_PROMPT
    elif chat_template == "dolphin":
        TEMPLATE = DOLPHIN_PROMPT
    elif chat_template == "llama3_cot" or chat_template == "llama3-cot":
        TEMPLATE = LLAMA3_COT_PROMPT
    elif chat_template == "llama3_cot_defender":
        # TEMPLATE = LLAMA3_COT_DEFENDER_PROMPT
        pass
    elif chat_template == "deepseek_r1_distill_llama3":
        TEMPLATE = DEEPSEEK_R1_DISTILL_LLAMA3_PROMPT
    elif chat_template == "game_defender":
        # Use model's own chat_template.jinja instead of hardcoded template
        # This ensures consistency with training-time template
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            # GAME_DEFENDER uses system prompt + user question format
            template = [
                {'role': 'system', 'content': GAME_DEFENDER_SYSTEM},
                {'role': 'user', 'content': DEFENDER_FIRST_USER_PROMPT}
            ]
            prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
            # Remove BOS token if present
            if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
                prompt = prompt.replace(tokenizer.bos_token, "")
            TEMPLATE = {
                'description': f"Template used by {model_name_or_path} (game_defender with tokenizer.apply_chat_template)",
                'prompt': prompt
            }
        except Exception as e:
            # Fallback to hardcoded Qwen template if tokenizer fails
            print(f"WARNING: Failed to load tokenizer for game_defender, using fallback Qwen template. Error: {e}")
            TEMPLATE = GAME_DEFENDER_PROMPT
    elif chat_template == "hf" or chat_template is None:
        # ======== Else default to tokenizer.apply_chat_template =======
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            template = [{'role': 'system', 'content': system_message},
                        {'role': 'user', 'content': '{instruction}'}] if system_message else [
                {'role': 'user', 'content': '{instruction}'}]
            prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
            # Check if the prompt starts with the BOS token
            # removed <s> if it exist (LlamaTokenizer class usually have this) as our baselines will add these if needed later
            if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
                prompt = prompt.replace(tokenizer.bos_token, "")
            TEMPLATE = {'description': f"Template used by {model_name_or_path} (tokenizer.apply_chat_template)",
                        'prompt': prompt}
        except:
            assert TEMPLATE, f"Can't find instruction template for {model_name_or_path}, and apply_chat_template failed."
    else:
        raise ValueError(f"Can't find chat template `{chat_template}`.")

    print("Found Instruction template for", model_name_or_path)
    print(TEMPLATE)

    return TEMPLATE


def _get_fschat_conv(model_name_or_path=None, fschat_template=None, system_message=None, **kwargs):
    template_name = fschat_template
    if template_name is None:
        template_name = model_name_or_path
        print(f"WARNING: default to fschat_template={template_name} for model {model_name_or_path}")
        template = get_conversation_template(template_name)
    else:
        template = get_conv_template(template_name)

    # New Fschat version remove llama-2 system prompt: https://github.com/lm-sys/FastChat/blob/722ab0299fd10221fa4686267fe068a688bacd4c/fastchat/conversation.py#L1410
    if template.name == 'llama-2' and system_message is None:
        print("WARNING: using llama-2 template without safety system promp")

    if system_message:
        template.set_system_message(system_message)

    assert template, "Can't find fschat conversation template `{template_name}`. See https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py for supported template"
    return template
