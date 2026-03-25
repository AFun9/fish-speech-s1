"""
音色特征提取和预编码脚本
提前提取音色特征并保存到磁盘，避免重启后重新编码
"""
 
import argparse
import torch
import ormsgpack
from pathlib import Path
from loguru import logger
 
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.file import read_ref_text
 
 
def extract_and_save_features(
    decoder_model, 
    engine: TTSInferenceEngine,
    audio_path: str,
    text: str,
    output_path: str
):
    """
    提取音频特征并保存到磁盘
    """
    logger.info(f"处理音频: {audio_path}")
    
    # 读取音频并提取特征
    prompt_tokens = engine.encode_reference(
        reference_audio=audio_path,
        enable_reference_audio=True
    )
    
    # 准备保存的数据
    save_data = {
        "prompt_tokens": prompt_tokens.cpu().numpy().tolist() if prompt_tokens is not None else None,
        "text": text,
        "audio_shape": list(prompt_tokens.shape) if prompt_tokens is not None else None
    }
    
    # 保存到磁盘
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用 ormsgpack 保存（与 API 保持一致）
    with open(output_file, "wb") as f:
        f.write(ormsgpack.packb(save_data))
    
    logger.info(f"特征已保存到: {output_path}")
    
    return save_data
 
 
def batch_extract_features(
    decoder_model,
    engine: TTSInferenceEngine,
    reference_dir: str,
    output_dir: str
):
    """
    批量提取 references/ 目录下所有音色特征
    """
    ref_path = Path(reference_dir)
    out_path = Path(output_dir)
    
    if not ref_path.exists():
        logger.error(f"音色目录不存在: {reference_dir}")
        return
    
    # 遍历所有音色文件夹
    for voice_dir in ref_path.iterdir():
        if not voice_dir.is_dir():
            continue
            
        logger.info(f"处理音色: {voice_dir.name}")
        
        # 查找音频文件
        audio_files = list(voice_dir.glob("*.wav")) + list(voice_dir.glob("*.mp3"))
        
        for audio_file in audio_files:
            # 查找对应的文本文件
            lab_file = audio_file.with_suffix(".lab")
            
            if not lab_file.exists():
                logger.warning(f"未找到文本文件: {lab_file}，跳过")
                continue
            
            # 读取文本
            text = read_ref_text(str(lab_file))
            
            # 输出文件路径
            output_file = out_path / voice_dir.name / f"{audio_file.stem}.tokens"
            
            # 提取并保存特征
            extract_and_save_features(
                decoder_model,
                engine,
                str(audio_file),
                text,
                str(output_file)
            )
 
 
def load_cached_features(cache_path: str):
    """
    从磁盘加载预提取的特征
    """
    with open(cache_path, "rb") as f:
        data = ormsgpack.unpackb(f.read())
    
    # 转换回 torch tensor
    if data["prompt_tokens"] is not None:
        data["prompt_tokens"] = torch.tensor(data["prompt_tokens"])
    
    return data
 
 
def main():
    parser = argparse.ArgumentParser(description="提取音色特征并保存到磁盘")
    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        default="checkpoints/openaudio-s1-mini",
        help="LLaMA 模型路径"
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/openaudio-s1-mini/codec.pth",
        help="解码器模型路径"
    )
    parser.add_argument(
        "--decoder-config-name",
        type=str,
        default="modded_dac_vq",
        help="解码器配置名称"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="使用半精度"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="references",
        help="音色目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reference_cache",
        help="特征缓存输出目录"
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="单个音频文件路径"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="音频对应的文本（单文件模式使用）"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    logger.info("加载解码器模型...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device
    )
    
    logger.info("加载 LLaMA 模型...")
    precision = torch.half if args.half else torch.bfloat16
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=precision,
        compile=False
    )
    
    logger.info("初始化推理引擎...")
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=precision,
        compile=False
    )
    
    # 提取特征
    if args.single:
        # 单文件模式
        if not args.text:
            logger.error("单文件模式需要提供 --text 参数")
            return
        
        output_file = Path(args.output_dir) / Path(args.single).stem / "tokens.cache"
        extract_and_save_features(
            decoder_model,
            engine,
            args.single,
            args.text,
            str(output_file)
        )
    else:
        # 批量模式
        batch_extract_features(
            decoder_model,
            engine,
            args.reference_dir,
            args.output_dir
        )
    
    logger.info("特征提取完成！")
 
 
if __name__ == "__main__":
    main()

"""

python export_pt.py \
  --llama-checkpoint-path checkpoints/openaudio-s1-mini \
  --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \
  --device cuda \
  --single assert/andelie.wav \
  --text "Здравствуйте, я Ханна, ваш автомобильный АИ помощник.Я помогу узнать погоду, построить маршрут, включить музыку и сделать поездку удобнее и безопаснее.Всегда рядом, чтобы помочь вам в дороге." \
  --output-dir reference_cache
"""