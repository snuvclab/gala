# xFormers
git clone https://github.com/facebookresearch/xformers.git
cd xformers
echo "numpy" > requirements.txt
git submodule update --init --recursive
python setup.py install
cd ..
rm -rf xformers

# Grounded SAM
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
cd GroundingDINO
python setup.py build
python setup.py install
cd ..
cd ..
mkdir segmentation/segment_anything
mkdir segmentation/GroundingDINO
mv Grounded-Segment-Anything/segment_anything/* segmentation/segment_anything
mv Grounded-Segment-Anything/GroundingDINO/* segmentation/GroundingDINO
cd segmentation
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
rm -rf Grounded-Segment-Anything


# SMPL-X
# Script from https://github.com/yfeng95/SCARF/blob/main/fetch_data.sh
mkdir -p ./deformer/data
# SMPL-X 2020 (neutral SMPL-X model with the FLAME 2020 expression blendshapes)
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O './deformer/data/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue

# scarf utilities
echo -e "\nDownloading data..."
wget https://owncloud.tuebingen.mpg.de/index.php/s/n58Fzbzz7Ei9x2W/download -O ./deformer/data/scarf_utilities.zip
unzip ./deformer/data/scarf_utilities.zip -d ./deformer/data
rm ./deformer/data/scarf_utilities.zip


# Generate tetrahedral grid
python utils/generate_tets.py
rm cube.obj