from ..base_attack import BaseAttack
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseWhiteBoxAttack(BaseAttack):
    """
    白盒攻击的基类，提供对模型内部访问的支持。
    
    白盒攻击需要访问模型的内部状态，如梯度、权重、嵌入层等。
    这个基类定义了白盒攻击的通用接口。
    """
    
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        
        # 验证模型是否支持白盒攻击所需的接口
        self._validate_model_capabilities()
    
    def _validate_model_capabilities(self):
        """验证模型是否支持白盒攻击所需的能力"""
        required_methods = ['get_input_embeddings', 'get_model', 'get_tokenizer']
        
        for method in required_methods:
            if not hasattr(self.model, method):
                raise ValueError(f"Model must support {method}() method for white-box attacks")
    
    @abstractmethod
    def attack(self, target: Any) -> Any:
        """
        执行白盒攻击
        
        Args:
            target: 攻击目标
            
        Returns:
            AttackResult: 攻击结果
        """
        pass
    
    def get_model_device(self):
        """获取模型所在的设备"""
        if hasattr(self.model, 'get_model'):
            return self.model.get_model().device
        return next(self.model.get_model().parameters()).device