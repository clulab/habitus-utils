#steps to run

- get conda

```
conda create --name predictables python=3
conda activate predictables
pip install -r requirements.txt
cd predictables/CGAP
mkdir log
python predict_withv2_dataobjects.py
 ```
 