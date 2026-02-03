# evaluation/models/qwen_api_model.py

import os
from typing import List
from openai import OpenAI
from evaluation.schemas import GeneratorModelBase


class QwenAPIModel(GeneratorModelBase):
    """适配Qwen3Guard API（OpenAI格式）"""
    
    def __init__(
        self,
        model_name_or_path: str,
        model_input_template_path_or_name: str = None,
        **kwargs
    ):
        # 传入参数给父类
        super().__init__(
            model_name_or_path=model_name_or_path,
            model_input_template_path_or_name=model_input_template_path_or_name
        )
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=os.environ.get(
                "WILDGUARD_API_ENDPOINT",
                'http://s-20251119153749-lp69w-decode.ailab-safethm.svc:23344/v1'
            ),
            api_key=os.environ.get("WILDGUARD_API_KEY", "FAKE_API_KEY"),
            timeout=30.0  # 30秒超时
        )
        
        # 加载Qwen模板
        from transformers import AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except:
            print(f"[QwenAPI] 无法加载本地tokenizer，使用默认Qwen格式")
            self.tokenizer = None
    
    def load_model(self):
        """加载模型（API模式下无需本地模型）"""
        pass  # OpenAI API客户端已在__init__初始化
    
    def generate_completions(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """生成补全文本"""
        return self.generate(prompts, **kwargs)
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        """批量生成（逐个调用API）"""
        
        responses = []
        for prompt in prompts:
            try:
                # 确保prompt是字符串
                if not isinstance(prompt, str):
                    prompt = str(prompt)
                
                messages = [{"role": "user", "content": prompt}]
                
                if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    # 确保formatted是字符串
                    if not isinstance(formatted, str):
                        formatted = str(formatted)
                    
                    completion = self.client.completions.create(
                        model="orm",
                        prompt=formatted,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    response_text = completion.choices[0].text
                else:
                    completion = self.client.chat.completions.create(
                        model="orm",
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    response_text = completion.choices[0].message.content
                
                responses.append(response_text)
                
            except Exception as e:
                print(f"[QwenAPI] 生成失败: {e}")
                responses.append("")
        
        return responses
    
    def required_input_fields(self) -> List[str]:
        return ["instruction"]
