echo "****************** Installing pytorch ******************"
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo ""
echo "****************** Installing numpy/scipy ******************"
pip install numpy==1.26.4 scipy

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
pip install matplotlib

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

echo ""
echo ""
echo "****************** Installing yaml tools ******************"
pip install PyYAML easydict

echo ""
echo ""
echo "****************** Installing tracking core deps ******************"
conda install -c conda-forge -y lmdb pycocotools libjpeg-turbo

echo ""
echo ""
echo "****************** Installing jpeg4py (optional fast loader) ******************"
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing timm / yacs / einops ******************"
pip install timm yacs einops

echo ""
echo ""
echo "****************** Installing transformers ******************"
pip install transformers==4.41.0

echo ""
echo ""
echo "****************** Installing training utilities ******************"
pip install ninja tensorboardX debugpy wandb setuptools==70.0.0

echo ""
echo ""
echo "****************** Installing extra tools ******************"
pip install colorama visdom scikit-image

echo ""
echo ""
echo "****************** Installation complete! ******************"
echo "Activate env: conda activate sfdatrack"