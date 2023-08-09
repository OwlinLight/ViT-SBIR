# ViT-SBIR
## Overview
This project aims to improve the performance of a Siamese Neural Network by incorporating Vision Transformer (ViT) layers into its architecture. The enhanced model was implemented and trained using PyTorch on a GPU of a cloud server. The inclusion of ViT layers led to a remarkable 4% increase in top-1 accuracy compared to the conventional approach.

Additionally, an AI-powered web application was built based on the algorithm. This web application allows users to sketch shoes on a drawing canvas, and it fetches the most matched product image from the library stored on the server. The application was developed using Flask, a lightweight Python web framework.
## Video Demo
:tv:[Watch the vid](https://youtu.be/CERm_XM1aTc)

## Prerequisites
- Python 3.6+
- PyTorch 1.7.1+
- Flask 1.1.2+

## Usage
### Training
To train the model, run the following command:
```bash
cd ViT-FGSBIR
python main.py --dataset_name ShoeV2 \
                    --backbone_name ViT \
                    --pool_method AdaptiveAvgPool2d \
                    --root_dir ./../ \
                    --batchsize 16 \
                    --nThreads 1 \
                    --learning_rate 0.0001 \
                    --max_epoch 100 \
                    --eval_freq_iter 100 \
                    --print_freq_iter 1
```
You can adjust the parameter as you wish

### Testing
download the Dataset from [here](https://drive.google.com/file/d/16chkeQCqln2rLKLFBQGc6yKlYdJnrmJm/view?usp=drive_link) and 
download the trained model from [here](https://drive.google.com/file/d/1KvImiQz3RWMkdapdufWsvYbM7Vk9aCTH/view?usp=drive_link) and put it in the folder `ViT-FGSBIR/`
To test the model, run the following command:
```bash
cd ViT-FGSBIR
python eval2.py
```

### Web Application
To run the web application, run the following command:
```bash
cd flaskr
export FLASK_APP=flaskr
export FLASK_ENV=development
flask run
```
Then, open http://localhost:5000/ in a browser.


## Disclaimer
This project is for educational and research purposes only. The AI web application may not be fully optimized for production use, and its performance may vary based on hardware and server capabilities.

## Contributer

The FG-SBIR pytorch code is developed based on [FG-SBIR](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yu_Sketch_Me_That_CVPR_2016_paper.pdf) and [FG-SBIR-PyTorch](https://github.com/AyanKumarBhunia/Baseline_FGSBIR)
The Flask application is developed based on [Flaskr](https://flask.palletsprojects.com/en/1.1.x/tutorial/)

## Contact
For any questions, issues, or collaborations, please contact [201806040620@zjut.edu.cn].