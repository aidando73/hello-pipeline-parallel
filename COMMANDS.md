```bash
source ~/miniconda3/bin/activate
conda create --prefix ./env
source ~/miniconda3/bin/activate ./env

pip install --no-cache-dir -r requirements.txt

git clone https://github.com/pytorch/PiPPy.git
cd PiPPy
python setup.py install
cd ..

```