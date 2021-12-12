CONDA_ENV_NAME=m2t_d2_demo

echo "create conda environment '${CONDA_ENV_NAME}'"
conda env create -n ${CONDA_ENV_NAME} -f environment.yml

echo "activate conda environnment '${CONDA_ENV_NAME}'"
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

echo "install other packages"
pip install opencv-python
python -m spacy download en

echo "install detectron2"
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@ffff8ac'

echo "cloning grid-feats-vqa"
git clone https://github.com/facebookresearch/grid-feats-vqa

echo "download & extract visual_genome.tgz"
mkdir -p datasets/visual_genome/annotations
TOPDIR=${PWD}
cd datasets/visual_genome/annotations
wget -c https://dl.fbaipublicfiles.com/grid-feats-vqa/json/visual_genome.tgz
tar zxf visual_genome.tgz
cd ${TOPDIR}

echo "cloning meshed-memory-transformer"
git clone https://github.com/fujie-cit/meshed-memory-transformer.git --branch dev-fujie

echo "download parameter files"
wget -c https://gist.githubusercontent.com/kumeS/6f67b4b9085b3d2580c51b8ae4953beb/raw/4587c1306e00a576037379ee410d6c35f3daada6/gdrive_download.sh
mkdir -p models
cd models
source ../gdrive_download.sh
if [ ! -e vocab_m2_transformer_sc_d2.pkl ]; then
    gdrive_download 1b-qzZ2IX1GfIEyzyAXHqFYV_AhvXFdu1 vocab_m2_transformer_sc_d2.pkl
fi
if [ ! -e m2_transformer_sc_d2_best.pth ]; then
    gdrive_download 1AJbGsIM9Qorbk_7DGRgNQWlvL6CCm1mi m2_transformer_sc_d2_best.pth
fi

