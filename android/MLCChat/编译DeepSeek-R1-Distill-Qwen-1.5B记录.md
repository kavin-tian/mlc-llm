# 编译 DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC 模型记录

本文档记录编译 `DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC` 模型的完整步骤和关键信息。

**模型地址**: https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC

**参考文档**: `BUILD_WINDOWS_ANDROID.md`

---

## 一、在 Ubuntu 22.04（VM/WSL2）编译模型库 .tar

### 1. 准备环境（如果还没有）

```bash
sudo apt update
sudo apt install -y git git-lfs build-essential cmake ninja-build clang curl zstd
git lfs install

# Miniconda
wget -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda create -n mlc-android python=3.13 -y && conda activate mlc-android

python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
```

### 2. 克隆模型权重

```bash
cd ~/Desktop/LLM  # 或你的工作目录
git lfs install
git clone https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC
```

### 3. 编译为 Android 模型库 .tar

```bash
python -m mlc_llm compile ~/Desktop/LLM/DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC \
  --device android \
  --overrides "prefill_chunk_size=128;context_window_size=2048" \
  --output ~/deepseek-r1-qwen1p5b-android.tar
```

### 4. 编译结果分析

**编译时间**: 2025-11-11 17:56:22

**关键信息**:
- **System-lib-prefix**: `qwen2_q4f16_1_`（注意结尾有下划线）
- **模型类型**: `qwen2`
- **量化方式**: `q4f16_1` (GroupQuantize)
- **输出文件**: `/home/kavin/deepseek-r1-qwen1p5b-android.tar`

**内存使用情况**:
- 无 KV cache: **1039.00 MB**
  - 参数: 953.50 MB
  - 临时缓冲区: 85.50 MB
- KV cache: **0.03 MB per token**
- 4K KV cache 总内存: **1151.00 MB**

**编译日志关键提示**:
```
WARNING auto_target.py:378: --system-lib-prefix is automatically picked from the filename, qwen2_q4f16_1_, this allows us to use the filename as the model_lib in android/iOS builds. Please avoid renaming the .tar file when uploading the prebuilt.
```

### 5. 将生成的 .tar 拷贝到 Windows

```bash
# 如果使用 WSL2，可以这样拷贝：
mkdir -p /mnt/c/Users/kavin/Desktop/LLM/mlc-llm/android/MLCChat/dist/lib
cp ~/deepseek-r1-qwen1p5b-android.tar /mnt/c/Users/kavin/Desktop/LLM/mlc-llm/android/MLCChat/dist/lib/
```

---

## 二、在 Windows 配置应用并打包

### 1. 配置 `mlc-package-config.json`

在 `C:\Users\kavin\Desktop\LLM\mlc-llm\android\MLCChat\` 目录下，编辑或创建 `mlc-package-config.json`：

```json
{
  "device": "android",
  "model_list": [
    {
      "model": "HF://mlc-ai/DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC",
      "model_id": "DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC",
      "estimated_vram_bytes": 1200000000,
      "model_lib": "qwen2_q4f16_1",
      "overrides": { 
        "prefill_chunk_size": 128, 
        "context_window_size": 2048 
      }
    }
  ],
  "model_lib_path_for_prepare_libs": {
    "qwen2_q4f16_1": "dist/lib/deepseek-r1-qwen1p5b-android.tar"
  }
}
```

**重要说明**:
- `model_lib` 必须使用 `"qwen2_q4f16_1"`（去掉编译日志中 `qwen2_q4f16_1_` 的结尾下划线）
- `estimated_vram_bytes` 设置为 1200000000（约 1.2GB）或 1300000000（约 1.3GB，更安全）
  - 基于编译日志: 4K KV cache 总内存为 1151 MB = 1,206,910,976 字节
  - 建议值: 1,200,000,000 (1144 MB) 或 1,300,000,000 (1240 MB，更推荐)
- `.tar` 文件路径必须与实际拷贝后的路径一致

### 2. 初始化 Windows 环境（每次新开 cmd 都需要）

```cmd
:: 1) 激活 conda 环境
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mlc-android

:: 2) 载入 MSVC 工具链
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=%PATH:C:\Users\kavin\miniconda3\envs\mlc-android\Library\mingw64\bin;=%"
set "PATH=%PATH:C:\Users\kavin\miniconda3\envs\mlc-android\Library\usr\bin;=%"
set LINK=

:: 3) 设置环境变量
set JAVA_HOME=D:\tools\jdks\ms-17.0.16
set ANDROID_NDK=D:\tools\SDK\ndk\27.0.11718014
set TVM_NDK_CC=%ANDROID_NDK%\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android24-clang.cmd
set TVM_SOURCE_DIR=C:\Users\kavin\Desktop\LLM\mlc-llm\3rdparty\tvm
set MLC_LLM_SOURCE_DIR=C:\Users\kavin\Desktop\LLM\mlc-llm
set MLC_JIT_POLICY=READONLY
```

### 3. 只读打包（使用已有 .tar，禁止 JIT）

```cmd
cd /d C:\Users\kavin\Desktop\LLM\mlc-llm\android\MLCChat
set MLC_JIT_POLICY=READONLY
python -m mlc_llm package
```

### 4. 构建并安装 APK

```cmd
gradlew.bat assembleDebug
adb devices
gradlew.bat installDebug
```

---

## 三、关键点说明

### 为什么 `model_lib` 是 `"qwen2_q4f16_1"`？

根据编译日志和文档说明：

1. **编译时生成的 system-lib-prefix**: `qwen2_q4f16_1_`（带结尾下划线 `_`）
2. **配置文件中需要去掉结尾下划线**: 因此 `model_lib` 应填写为 `"qwen2_q4f16_1"`

**规则总结**:
- System-lib-prefix 带下划线用于内部符号命名（如 `qwen2_q4f16_1_function_name`）
- `model_lib` 作为标识符，去掉下划线更简洁，且系统会自动匹配

**文档依据**:
> 编译日志会显示自动推断的 `--system-lib-prefix`，如 `phi3_q4f16_0_`。在 `mlc-package-config.json` 中应填写 `model_lib: "phi3_q4f16_0"`（去掉末尾下划线），并保持 `.tar` 文件名与日志推断前缀的一致性（不要随意改名）。

### 如何估算 `estimated_vram_bytes`？

`estimated_vram_bytes` 用于告诉应用该模型需要多少 VRAM（显存/内存），帮助应用判断设备是否有足够的内存来加载模型。

**估算方法**:

1. **查看编译日志中的内存使用信息**:
   ```
   [2025-11-11 17:56:35] INFO model_metadata.py:94: Total memory usage without KV cache: 1039.00 MB (Parameters: 953.50 MB. Temporary buffer: 85.50 MB)
   [2025-11-11 17:56:35] INFO model_metadata.py:128: KV cache size: 0.03 MB per token in the context window
   [2025-11-11 17:56:35] INFO model_metadata.py:133: Total memory usage with a 4K KV cache: 1151.00 MB
   ```

2. **基于 "Total memory usage with a 4K KV cache" 值进行估算**:
   - 编译日志显示: **1151.00 MB** (4K KV cache 总内存)
   - 转换为字节: 1151 × 1024 × 1024 = **1,206,910,976 字节**
   - 建议值: **1,200,000,000 字节** (约 1.2GB) 或 **1,300,000,000 字节** (约 1.24GB，更安全)

3. **计算公式**:
   ```
   estimated_vram_bytes = (编译日志中的总内存MB) × 1024 × 1024 × 安全系数
   ```
   - 安全系数建议: 1.0 - 1.15（留 0-15% 余量）

4. **对于本模型**:
   - 基础值: 1151 MB = 1,206,910,976 字节
   - 建议值: **1,200,000,000** (1144 MB) 或 **1,300,000,000** (1240 MB，更推荐)
   - 当前配置: 1,200,000,000 (约 1.2GB)

**注意事项**:
- 这个值主要用于应用层的内存管理，不是硬性限制
- 如果设备内存充足，可以设置稍大一些（如 1.3GB），更安全
- 如果设备内存紧张，可以设置稍小一些，但不应小于编译日志中的总内存值

**参考其他模型的设置** (来自 `test.json`):
- Qwen3-0.6B: 3,000,000,000 (3GB)
- Phi-3.5-mini: 4,250,586,449 (约 4.05GB)
- Llama-3.2-3B: 4,679,979,417 (约 4.46GB)
- Mistral-7B: 4,115,131,883 (约 3.92GB)

### 其他注意事项

1. **不要随意重命名 .tar 文件**: 文件名与 system-lib-prefix 有关联，随意改名可能导致匹配失败
2. **确保路径一致**: `model_lib_path_for_prepare_libs` 中的路径必须与实际文件位置一致
3. **内存估算**: 根据编译日志，建议 `estimated_vram_bytes` 设置为 1,200,000,000（1.2GB）或 1,300,000,000（1.3GB，更安全）

---

## 四、完整编译日志（参考）

```
[2025-11-11 17:56:22] INFO auto_config.py:70: Found model configuration: /home/kavin/Desktop/LLM/DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC/mlc-chat-config.json
[2025-11-11 17:56:22] INFO auto_config.py:154: Found model type: qwen2. Use `--model-type` to override.
[2025-11-11 17:56:22] WARNING auto_target.py:378: --system-lib-prefix is automatically picked from the filename, qwen2_q4f16_1_, this allows us to use the filename as the model_lib in android/iOS builds. Please avoid renaming the .tar file when uploading the prebuilt.

Compiling with arguments:
  --config          QWen2Config(hidden_act='silu', hidden_size=1536, intermediate_size=8960, num_attention_heads=12, num_hidden_layers=28, num_key_value_heads=2, rms_norm_eps=1e-06, rope_theta=10000, vocab_size=151936, tie_word_embeddings=False, context_window_size=131072, prefill_chunk_size=8192, tensor_parallel_shards=1, head_dim=128, dtype='float32', max_batch_size=128, kwargs={})
  --quantization    GroupQuantize(name='q4f16_1', kind='group-quant', group_size=32, quantize_dtype='int4', storage_dtype='uint32', model_dtype='float16', linear_weight_layout='NK', quantize_embedding=True, quantize_final_fc=True, num_elem_per_storage=8, num_storage_per_group=4, max_int_value=7, tensor_parallel_shards=0)
  --model-type      qwen2
  --target          {'kind': 'opencl', 'tag': '', 'keys': ['opencl', 'gpu'], 'host': {'kind': 'llvm', 'tag': '', 'keys': ['arm_cpu', 'cpu'], 'mtriple': 'aarch64-linux-android'}, 'max_threads_per_block': 256, 'max_shared_memory_per_block': 16384, 'max_num_threads': 256, 'thread_warp_size': 1, 'texture_spatial_limit': 16384, 'max_function_args': 128, 'image_base_address_alignment': 64}
  --opt             flashinfer=0;cublas_gemm=0;faster_transformer=0;cudagraph=0;cutlass=0;ipc_allreduce_strategy=NONE
  --system-lib-prefix "qwen2_q4f16_1_"
  --output          /home/kavin/deepseek-r1-qwen1p5b-android.tar
  --overrides       context_window_size=2048;sliding_window_size=None;prefill_chunk_size=128;attention_sink_size=None;max_batch_size=None;tensor_parallel_shards=None;pipeline_parallel_stages=None;disaggregation=None

[2025-11-11 17:56:22] INFO config.py:107: Overriding context_window_size from 131072 to 2048
[2025-11-11 17:56:22] INFO config.py:107: Overriding prefill_chunk_size from 8192 to 128
[2025-11-11 17:56:35] INFO model_metadata.py:94: Total memory usage without KV cache: 1039.00 MB (Parameters: 953.50 MB. Temporary buffer: 85.50 MB)
[2025-11-11 17:56:35] INFO model_metadata.py:128: KV cache size: 0.03 MB per token in the context window
[2025-11-11 17:56:35] INFO model_metadata.py:133: Total memory usage with a 4K KV cache: 1151.00 MB
[2025-11-11 17:56:35] INFO compile.py:208: Generated: /home/kavin/deepseek-r1-qwen1p5b-android.tar
```

---

## 得到deepseek-r1-qwen1p5b-android.tar后,在windows上编译常见问题:
mlc-package-config.json 填写错误的名字: 
错误: "model_lib": "deepseek_r1_qwen1p5b_q4f16_1"
正确:  "model_lib": "qwen2_q4f16_1" (从日志中获取的,不是随便猜的)
下面是报错的日志:
```
100%|██████████████████████████████████████████████████████████████████████████████████| 31/31 [06:25<00:00, 12.45s/it]
[2025-11-11 18:13:28] INFO download_cache.py:197: Moving C:\Users\kavin\AppData\Local\Temp\tmppqo35loz\tmp to C:\Users\kavin\AppData\Local\mlc_llm\model_weights\hf\mlc-ai\DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC
[2025-11-11 18:13:28] INFO package.py:154: Dump the app config below to "dist\bundle\mlc-app-config.json":
{
  "model_list": [
    {
      "model_id": "Qwen3-0.6B-q0f16-MLC",
      "model_lib": "qwen3_q0f16",
      "model_url": "https://huggingface.co/mlc-ai/Qwen3-0.6B-q0f16-MLC",
      "estimated_vram_bytes": 3000000000
    },
    {
      "model_id": "Phi-3.5-mini-instruct-q4f16_0-MLC",
      "model_lib": "phi3_q4f16_0",
      "model_url": "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_0-MLC",
      "estimated_vram_bytes": 4250586449
    },
    {
      "model_id": "DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC",
      "model_lib": "deepseek_r1_qwen1p5b_q4f16_1",
      "model_url": "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC",
      "estimated_vram_bytes": 2000000000
    }
  ]
}
[2025-11-11 18:13:28] INFO package.py:211: Creating lib from ['dist/lib/qwen3-0p6b-android.tar', 'dist/lib/phi3p5-mini-android.tar', 'dist/lib/deepseek-r1-qwen1p5b-android.tar']
[2025-11-11 18:13:28] INFO package.py:212: Validating the library dist\lib\libmodel_android.a
[2025-11-11 18:13:28] INFO package.py:213: List of available model libs packaged: ['qwen3_q0f16', 'phi3_q4f16_0', 'qwen2_q4f16_1'], if we have '-' in the model_lib string, it will be turned into '_'
[2025-11-11 18:13:28] INFO package.py:252: ValidationError:
        model_lib deepseek_r1_qwen1p5b_q4f16_1 requested in dist\bundle\mlc-app-config.json is not found in dist\lib\libmodel_android.a
        specifically the model_lib for dist/lib/deepseek-r1-qwen1p5b-android.tar.
        current available model_libs in dist\lib\libmodel_android.a: ['qwen3_q0f16', 'phi3_q4f16_0', 'qwen2_q4f16_1']
        This can happen when we manually specified model_lib_path_for_prepare_libs in mlc-package-config.json
        Consider remove model_lib_path_for_prepare_libs (so library can be jitted)or check the compile command
[2025-11-11 18:13:28] INFO package.py:258: Validation failed
```


## 五、参考资源

- **模型地址**: https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC
- **构建文档**: `BUILD_WINDOWS_ANDROID.md`
- **官方文档**: https://llm.mlc.ai/docs/deploy/android.html

---

**创建时间**: 2025-11-11  
**最后更新**: 2025-11-11

