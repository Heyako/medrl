---如果安装了数据盘--
---数据盘迁移-
mkdir -p /data
mount /dev/vdb /data
---数据盘初始化---
lsblk
vdb  253:16   0    50G  0 disk
mkfs.ext4 /dev/vdb
mkdir -p /data
mount /dev/vdb /data
# 创建环境目录
mkdir -p /data/conda_envs
conda create -p /data/conda_envs/medrl python=3.10 -y
conda activate /data/conda_envs/medrl
cd /data
git clone https://github.com/Heyako/medrl.git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.46.0 accelerate peft datasets deepspeed openai wandb tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
cd medrl
pip install -e .


-----其它情况---
cd medrl
nano setup_gpu_env.sh

在 GPU 机器上创建并运行这个脚本（从你的项目根目录执行）：

  #!/bin/bash
  # setup_gpu_env.sh — MedRL GPU 环境初始化
  set -e

  # ── 1. 安装 miniconda（如果没有）──
  if ! command -v conda &>/dev/null; then
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
      bash ~/miniconda.sh -b -p ~/miniconda
      eval "$(~/miniconda/bin/conda shell.bash hook)"
      conda init
  fi

  # ── 2. 创建 medrl 环境 (Python 3.10 + CUDA 12.4 PyTorch) ──
  conda create -n medrl python=3.10 -y
  conda activate medrl

  # ── 3. 安装 PyTorch (CUDA 12.4 对应 torch >= 2.4) ──
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

  # ── 4. 验证 PyTorch 能看见 GPU ──
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: 
  {torch.cuda.get_device_name(0)}'); print(f'VRAM: 
  {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"

  # ── 5. 安装核心依赖 ──
  pip install transformers==4.46.0 accelerate peft datasets
  pip install deepspeed
  pip install flash-attn --no-build-isolation   # FlashAttention-2
  pip install openai# Judge API 调用
  pip install wandb tensorboard# 日志/监控

  # ── 6. 安装 MedRL 项目本身 (可编辑模式) ──
  pip install -e .

chmod +x setup_gpu_env.sh
./setup_gpu_env.sh

export JUDGE_API_KEY="sk-39ee85ca5af245709a4d1868c6acb123"
export JUDGE_BASE_URL="https://api.deepseek.com/v1"  
export JUDGE_MODEL="deepseek-chat"

scp -P 14312 data/raw/medqa_us_train.jsonl root@223.109.239.32:~/medrl/data/raw/
git pull origin main

如果没有torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch; print(torch.__version__); print('GPU可用状态:', torch.cuda.is_available())"

其余依赖
pip install transformers==4.46.0 accelerate peft datasets deepspeed openai wandb tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
这个包容易报错
pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
挂载项目本身
pip install -e .

pip install transformers==4.46.0 accelerate peft datasets deepspeed openai wandb tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .