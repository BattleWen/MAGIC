import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_replay_buffer_data(source_dir: str, output_dir: str, max_steps: int = 500):
    """
    提取 replay buffer 中的 attacker-defender 数据
    将每一条对话单独提取为一个对象
    
    Args:
        source_dir: 源目录路径
        output_dir: 输出目录路径
        max_steps: 最多处理的步数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_records = 0
    processed_steps = 0
    
    # 遍历所有的 train_step_*.jsonl 文件
    for step in range(1, max_steps + 1):
        if(step%20!=0):
            continue
        jsonl_path = os.path.join(source_dir, f'train_step_{step}.jsonl')
        
        if not os.path.exists(jsonl_path):
            logger.warning(f"文件不存在: {jsonl_path}")
            continue
        
        logger.info(f"处理文件: {jsonl_path}")
        
        step_records = []
        step_total = 0
        
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # 验证必要的字段
                        if 'history' not in data or 'role_scores' not in data:
                            logger.warning(f"跳过不完整的记录 (step {step}, line {line_idx})")
                            continue
                        
                        question = data.get('question', '')
                        history = data.get('history', [])
                        role_scores = data.get('role_scores', [])
                        
                        # 提取 4 条 attacker-defender 记录
                        if len(history) != 4:
                            logger.warning(f"预期 4 条历史记录，实际有 {len(history)} 条 (step {step}, line {line_idx})")
                        
                        if len(role_scores) != 4:
                            logger.warning(f"预期 4 条得分，实际有 {len(role_scores)} 条 (step {step}, line {line_idx})")
                        
                        # 遍历 history 中的每一条对话
                        for exchange_idx, (exchange, score) in enumerate(zip(history, role_scores)):
                            attacker_data = {}
                            defender_data = {}
                            
                            # 处理 attacker 和 defender 的内容
                            for message in (exchange if isinstance(exchange, list) else [exchange]):
                                if isinstance(message, dict):
                                    role = message.get('role', '')
                                    if role == 'attacker':
                                        attacker_data = {
                                            'parsed_think': message.get('parsed_think', ''),
                                            'parsed_answer': message.get('parsed_answer', '')
                                        }
                                    elif role == 'defender':
                                        defender_data = {
                                            'parsed_think': message.get('parsed_think', ''),
                                            'parsed_answer': message.get('parsed_answer', '')
                                        }
                            
                            # 创建单独的对话记录
                            extracted_record = {
                                'question': question,
                                'step': step,
                                'exchange_id': exchange_idx,
                                'attacker': attacker_data,
                                'defender': defender_data,
                                'scores': score
                            }
                            
                            step_records.append(extracted_record)
                            step_total += 1
                            total_records += 1
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON 解析错误 (step {step}, line {line_idx}): {e}")
                        continue
        
        except Exception as e:
            logger.error(f"处理文件时出错 {jsonl_path}: {e}")
            continue
        
        # 为每个 step 保存单独的文件
        if step_records:
            output_file = os.path.join(output_dir, f'extracted_step_{step}.jsonl')
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in step_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"Step {step} 提取完成，保存 {step_total} 条记录到: {output_file}")
            processed_steps += 1
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, 'extraction_stats.json')
    stats = {
        'total_records_extracted': total_records,
        'processed_steps': processed_steps,
        'max_steps_processed': max_steps,
        'output_directory': output_dir
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"全部提取完成! 共处理 {processed_steps} 个 step，提取了 {total_records} 条记录")
    logger.info(f"统计信息已保存到: {stats_file}")

if __name__ == '__main__':
    # 配置路径
    model='v5-D-q257bi-A-q257bisft_wocode-only-ratio11-freq15-reward1_0.5_0-woDformat-wo_label_reward-revised_label-2026-01-20_23-49-00'
    source_directory = f'/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wenxiaoyu/game/checkpoints/Game-separated/{model}/replay_buffer'
    output_directory = f'/mnt/shared-storage-user/wenxiaoyu/game-private/eval/pattern/raw_json/{model}'
    
    # 执行提取
    extract_replay_buffer_data(source_directory, output_directory, max_steps=300)