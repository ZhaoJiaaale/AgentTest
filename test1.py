import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import gc
from datasets import load_dataset
from src.models.QwenVL import MLLM
from src.model_edit.util import compute_covariance_matrix, covariance_based_edit
from src.utils.logger import create_logger

logger = create_logger()

if __name__ == "__main__":
    """ 配置参数 """
    model_id = "lingshu-medical-mllm/Lingshu-7B"
    safe_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    edit_method = "ROME"

    target_layers_idx = [16, 17, 18]
    target_token_idx = 0

    SAVE_MODEL_PATH = f"exps/results/models/{model_id.split("/")[-1]}-Safety-Edited-{edit_method}-layer_{"-".join(str(layer) for layer in target_layers_idx)}-token_{target_token_idx}"
    
    # Harmful Dataset Selection
    selected_indices = [3, 7, 10, 14, 22, 35, 41, 43, 44, 45, 46, 48, 54, 71, 73, 75, 77, 78, 95, 98, 122, 144, 145, 147, 148, 152, 156, 162, 165, 178, 187, 202, 211, 212, 216, 221, 232, 240, 241, 243, 247, 262, 264, 266, 268, 269, 274, 280, 288, 293, 294, 295, 299, 302, 311, 317, 322, 325, 341, 350, 356, 361, 369, 373, 377, 378, 388, 407, 408, 411, 419, 420, 423, 424, 428, 455, 460, 478, 482, 495, 508, 509, 510, 512, 519, 524, 526, 531, 533, 534, 536, 553, 563, 565, 573, 575, 577, 589, 595, 619, 622, 631, 636, 646, 655, 661, 666, 667, 670, 675, 689, 694, 696, 704, 705, 706, 711, 717, 718, 720, 721, 723, 729, 732, 735, 739, 748, 759, 760, 761, 766, 769, 786, 790, 792, 794, 807, 816, 817, 821, 827, 833, 835, 841, 847, 848, 855, 856, 863, 865, 868, 869, 870, 873, 875, 881, 883, 890, 899, 902, 906, 925, 928, 934, 937, 941, 943, 944, 951]





# # 配置参数
# model_id = "./models/Lingshu-7B"
# safe_model_id = "./models/Qwen2.5-VL-7B-Instruct"
# device = "cuda:0"
# batch_size = 32
# # 实验超参数
# alpha = 3.2                # 安全向量外推系数 (Intensity of safety injection)
# lambda_reg = 100.0         # 正则化强度 (Constraints strength)
# n_harm = 10                # 攻击样本数量
# n_med_calc = 500           # 计算协方差用的医学样本数
# # 层与Token定位
# target_layer_idx = 24
# target_token_idx = 0
# safe_target_layer_idx = 24
# safe_target_token_idx = 0

# SAVE_MODEL_PATH = f"results/models/Lingshu-7B-Safety-Edited-ROME-alpha_{alpha}"

# # 数据加载
# harm_data = load_dataset("jialelele/med_mllm_safety_benchmark", split="catqa_figstep_train")
# harm_prompts = [item["jailbreak_prompt"] for item in harm_data]
# harm_images = [item["image"] for item in harm_data]

# # -----------------------------------------------------------------------------
# # Phase 1: 安全模型 (Qwen) 提取目标特征
# # -----------------------------------------------------------------------------
# logger.info(f"=== Phase 1: Extracting Safety Targets from Base Model: {safe_model_id} ===")

# # 1. 加载安全模型
# mllm_safe = MLLM(safe_model_id, device=device)

# # 2. 筛选样本 (获取 Sorry 和 Safe 的样本索引)
# # sorry_indices = get_sorry_indices(mllm=mllm_safe, prompts=harm_prompts, images=harm_images)
# # safe_indices = get_safe_indices(mllm=mllm_safe, prompts=harm_prompts, images=harm_images)
# # logger.info(f"Indices - Sorry: {sorry_indices}, Safe: {safe_indices}")

# # 3. 确定最终使用的攻击样本
# # selected_indices = list(set(sorry_indices) | set(safe_indices))[:n_harm]
# selected_indices = [3, 5, 18, 22, 37, 62, 70, 104, 111, 130]
# harm_prompts_selected = [harm_data[i]["jailbreak_prompt"] for i in selected_indices]
# harm_images_selected = [harm_data[i]["image"] for i in selected_indices]

# # 4. 提取安全目标向量 (Target Outputs)
# _, target_harm_out_ht = mllm_safe.extract_hidden_state(
#     prompts=harm_prompts_selected,
#     image_paths=harm_images_selected,
#     target_token_idx=safe_target_token_idx,
#     target_layer_idx=safe_target_layer_idx,
#     weight_matrix_name="down_proj",
#     batch_size=batch_size
# )
# target_harm_out_ht = target_harm_out_ht.cpu() # 移至内存

# # 5. 释放安全模型显存
# logger.info("Unloading Safe Model to free GPU memory...")
# mllm_safe.clear_memory()
# del mllm_safe
# gc.collect()
# torch.cuda.empty_cache()

# # -----------------------------------------------------------------------------
# # Phase 2: 医学模型 (Lingshu) 计算协方差并编辑
# # -----------------------------------------------------------------------------
# logger.info("=== Phase 2: Loading Medical Model for Editing ===")

# mllm = MLLM(model_id, device=device)

# # 1. 提取攻击样本在医学模型中的 Input (Keys) 和 Output (用于外推)
# harm_in_ht, harm_out_ht = mllm.extract_hidden_state(
#     prompts=harm_prompts_selected,
#     image_paths=harm_images_selected,
#     target_token_idx=target_token_idx,
#     target_layer_idx=target_layer_idx,
#     weight_matrix_name="down_proj",
#     batch_size=batch_size,
# )
# # 保持在 GPU 或 CPU 均可，这里先放 GPU 方便计算
# harm_in_ht = harm_in_ht.to(device)
# harm_out_ht = harm_out_ht.to(device)

# # 2. 构建最终编辑目标 (Vector Extrapolation)
# # 将 Phase 1 存的 CPU tensor 移回 GPU
# target_harm_out_ht = target_harm_out_ht.to(device)

# # Formula: Target = Med_Out + alpha * (Safe_Base_Out - Med_Out)
# target_harm_out_ht_final = harm_out_ht + (target_harm_out_ht - harm_out_ht) * alpha
# logger.info(f"Safety targets constructed with alpha={alpha}")

# # 3. 计算医学数据的协方差矩阵 (Covariance Calculation)
# med_dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")
# med_prompts = [item["question"] for item in med_dataset][:n_med_calc]
# med_images = [item["image"] for item in med_dataset][:n_med_calc]

# C_med = compute_covariance_matrix(
#     mllm=mllm,
#     prompts=med_prompts,
#     images=med_images,
#     target_layer_idx=target_layer_idx,
#     target_token_idx=target_token_idx,
#     batch_size=16
# )

# # 4. 执行编辑
# target_layer = mllm.llm.model.language_model.layers[target_layer_idx].mlp.down_proj
# W_old = target_layer.weight.data.clone()

# # 调用基于协方差的编辑函数
# W_new = covariance_based_edit(
#     W_old=W_old,
#     C=C_med,
#     K_harm=harm_in_ht,
#     V_target=target_harm_out_ht_final,
#     lambda_reg=lambda_reg
# )

# # 5. 应用权重
# with torch.no_grad():
#     target_layer.weight.data.copy_(W_new.half()) # 假设模型是 fp16/bf16

# # -----------------------------------------------------------------------------
# # Phase 3: 验证与保存
# # -----------------------------------------------------------------------------
# logger.info("=== Phase 3: Validation & Saving ===")

# # 保存模型
# logger.info(f"Saving to {SAVE_MODEL_PATH}...")
# mllm.llm.save_pretrained(SAVE_MODEL_PATH)
# mllm.processor.save_pretrained(SAVE_MODEL_PATH)
# logger.info("Done.")
