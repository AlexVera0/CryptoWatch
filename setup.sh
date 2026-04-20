#!/bin/bash
# ============================================================
# setup.sh — 一键环境安装（带进度输出，已装的自动跳过）
# 用法：bash setup.sh
# ============================================================

set -e
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC}   $1"; }
skip() { echo -e "  ${BLUE}[SKIP]${NC} $1（已安装）"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
info() { echo -e "  ${BLUE}....${NC}  $1"; }
section() { echo -e "\n${BLUE}══════════════════════════════════════${NC}"; echo -e "  $1"; echo -e "${BLUE}══════════════════════════════════════${NC}"; }

echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  CryptoWatch V4 — 环境安装程序${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"

# ---- 1. 系统检测 ----
section "1/6  系统环境检测"

OS=$(lsb_release -rs 2>/dev/null || echo "unknown")
ARCH=$(uname -m)
ok "操作系统: Ubuntu $OS ($ARCH)"

TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
FREE_MEM=$(free -m | awk '/^Mem:/{print $7}')
DISK_FREE=$(df -h . | awk 'NR==2{print $4}')
ok "总内存: ${TOTAL_MEM}MB | 可用: ${FREE_MEM}MB"
ok "磁盘剩余: $DISK_FREE"

PY_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
PY_MAJOR=$(echo $PY_VER | cut -d. -f1)
PY_MINOR=$(echo $PY_VER | cut -d. -f2)

if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
    ok "Python $PY_VER（兼容 ✓）"
else
    warn "Python $PY_VER 低于3.10，正在安装3.12..."
    apt-get update -q && apt-get install -y python3.12 python3.12-venv python3.12-dev -q
    ok "Python 3.12 安装完成"
fi

# ---- 2. 系统依赖 ----
section "2/6  系统依赖检测"

check_and_install_pkg() {
    local pkg=$1
    if dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        skip "$pkg"
    else
        info "安装 $pkg ..."
        apt-get install -y "$pkg" -q 2>/dev/null && ok "$pkg 安装完成" || warn "$pkg 安装失败（非关键）"
    fi
}

apt-get update -q 2>/dev/null

check_and_install_pkg "python3-pip"
check_and_install_pkg "python3-venv"
check_and_install_pkg "python3-dev"
check_and_install_pkg "build-essential"
check_and_install_pkg "p7zip-full"
check_and_install_pkg "screen"
check_and_install_pkg "curl"

# ---- 3. 虚拟环境 ----
section "3/6  Python 虚拟环境"

if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    skip "虚拟环境（venv/）"
else
    info "创建虚拟环境..."
    python3 -m venv venv
    ok "虚拟环境创建完成"
fi

source venv/bin/activate
ok "虚拟环境已激活"

# ---- 4. Python 依赖包 ----
section "4/6  Python 依赖包安装"

# 升级pip
if pip show pip | grep -q "Version"; then
    PIP_VER=$(pip --version | grep -oP '\d+\.\d+' | head -1)
    skip "pip $PIP_VER"
fi
pip install --upgrade pip -q

# 逐个检查并安装核心包
PACKAGES=(
    "pandas:pandas"
    "numpy:numpy"
    "xgboost:xgboost"
    "scikit-learn:sklearn"
    "pandas-ta:pandas_ta"
    "requests:requests"
    "filelock:filelock"
    "loguru:loguru"
    "plotly:plotly"
    "pyarrow:pyarrow"
    "python-dotenv:dotenv"
    "joblib:joblib"
    "scipy:scipy"
    "backtesting:backtesting"
    "schedule:schedule"
    "imbalanced-learn:imblearn"
)

NEED_INSTALL=()
for entry in "${PACKAGES[@]}"; do
    pkg_name="${entry%%:*}"
    import_name="${entry##*:}"
    if python3 -c "import $import_name" 2>/dev/null; then
        skip "$pkg_name"
    else
        warn "$pkg_name 未安装，加入安装队列"
        NEED_INSTALL+=("$pkg_name")
    fi
done

if [ ${#NEED_INSTALL[@]} -eq 0 ]; then
    ok "所有Python包已安装，跳过"
else
    echo ""
    info "开始安装 ${#NEED_INSTALL[@]} 个缺失的包..."
    echo "  安装列表: ${NEED_INSTALL[*]}"
    echo ""

    TOTAL=${#NEED_INSTALL[@]}
    CURRENT=0
    for pkg in "${NEED_INSTALL[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo -ne "  [${CURRENT}/${TOTAL}] 正在安装 ${pkg}..."
        if pip install "$pkg" -q 2>/dev/null; then
            echo -e " ${GREEN}完成${NC}"
        else
            echo -e " ${RED}失败${NC}"
            warn "尝试从 requirements.txt 完整安装..."
            pip install -r requirements.txt -q
            break
        fi
    done
    ok "Python依赖安装完成"
fi

# ---- 5. 配置文件 ----
section "5/6  配置文件检测"

if [ -f ".env" ]; then
    # 检查关键配置是否填写
    if grep -q "your_binance_api_key_here" .env 2>/dev/null; then
        warn ".env 存在但 BINANCE_API_KEY 未填写"
        echo "         请运行: nano .env"
    else
        ok ".env 文件已配置"
    fi
else
    cp .env.example .env
    warn ".env 文件不存在，已从模板创建"
    echo ""
    echo -e "  ${RED}请立即填写密钥（必须）：${NC}"
    echo "    nano .env"
    echo ""
    echo "  需要填写的5个项目："
    echo "    BINANCE_API_KEY=你的Binance API Key"
    echo "    BINANCE_API_SECRET=你的Binance API Secret"
    echo "    SMTP_PASSWORD=QQ邮箱16位授权码"
    echo "    ALERT_EMAIL_FROM=你的QQ@qq.com"
    echo "    ALERT_EMAIL_TO=接收信号的邮箱"
fi

python3 main.py --validate-config 2>/dev/null || true

# ---- 6. 模型文件检测 ----
section "6/6  AI模型文件检测"

MODEL_COUNT=$(ls models/*_model.pkl 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -gt 0 ]; then
    ok "已找到 $MODEL_COUNT 个训练好的模型文件"
    skip "训练步骤（模型已存在）"
else
    warn "未找到模型文件，需要先训练"
    echo ""
    echo -e "  ${YELLOW}请运行以下命令训练（约 1-2 分钟）：${NC}"
    echo "    source venv/bin/activate"
    echo "    python main.py --mode research --top-n 5 --backtest-only"
    echo ""
    echo "  训练完成后再运行："
    echo "    python startup.py"
fi

# ---- 完成 ----
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  安装完成！${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
echo "后续步骤："
if grep -q "your_binance_api_key_here" .env 2>/dev/null; then
    echo -e "  ${RED}[必须]${NC} 填写密钥:   nano .env"
fi
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo -e "  ${YELLOW}[必须]${NC} 训练模型:   source venv/bin/activate && python main.py --mode research --top-n 5 --backtest-only"
fi
echo -e "  ${GREEN}[启动]${NC} 启动监控:   source venv/bin/activate && python startup.py"
echo ""
