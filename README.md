# LLM-DiffRec

## Environment Requirements

- Anaconda 3
- Python 3.11.13
- PyTorch 2.5.1
- NumPy 1.24.0
- Pandas 2.3.3

## Setup Environment

**1. Install Dependencies:**
```bash
pip install torch pandas numpy scipy bottleneck tqdm sentence-transformers transformers
```

**2. Configure Hardware:**
- If you are using a GPU:
  ```bash
  bash setup_env_gpu.sh
  ```
- If you are using an NPU:
  ```bash
  bash setup_env_npu.sh
  ```

## Data Preparation

The experimental data is located in the `./datasets` folder (e.g., Amazon Musical Instruments and Amazon Software). Raw data can be downloaded from the [Amazon Review Data (2018) website](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).

You can use the processed data in ./datasets directly

Or you can preprocess the data from scratch:

1. **Download Data:** Download the 5-core `.json` data and metadata from the link provided above.
2. **Extract:** Unzip the files and place them in a subdirectory under `./datasets` (e.g., `./datasets/raw_amazon_instruments`).
3. **Preprocess:** Edit `preprocess_data.py`. Update `RAW_DIR` to your raw data path and `PROCESSED_DIR` to your target directory (e.g., `RAW_DIR=./raw_amazon_instruments`, `PROCESSED_DIR=./amazon-instruments`). Run the script.
4. **Generate Embeddings:** Edit `LLM-DiffRec/generate_embeddings.py` by updating `PROCESSED_DIR` and `EMBEDDING_DIR` to the correct paths. Then run:
   ```bash
   cd LLM-DiffRec
   python generate_embeddings.py
   ```
5. **Generate Cold-start Datasets:** Navigate to the datasets directory and run the data generation script:
   ```bash
   cd ../datasets
   python generate_coldstart_dataset.py
   ```
You have now finished the data processing part.

## Training

To train the models and fine-tune hyperparameters, navigate to the `LLM-DiffRec` directory:
```bash
cd ./LLM-DiffRec
```

Ensure that the hyperparameter `noise_min` is set to a value strictly lower than `noise_max`. Execute the bash script:
```bash
bash run_semantic_npu.sh <dataset> <lr> <weight_decay> <batch_size> <dims> <emb_size> <mean_type> <steps> <noise_scale> <noise_min> <noise_max> <sampling_steps> <reweight> <log_name> <round> <gpu_id> <use_semantic> <model_type> <semantic_dim> <semantic_proj_dim>
```

### Training Modes
Configure `<use_semantic>` and `<model_type>` depending on the desired model variant:
- **Original DiffRec:** `<use_semantic>=0`, `<model_type>=original`
- **Semantic:** `<use_semantic>=1`, `<model_type>=semantic`
- **Dual-Stream:** `<use_semantic>=1`, `<model_type>=dual`
- **FiLM:** `<use_semantic>=1`, `<model_type>=film`
- **FiLM-Dot:** `<use_semantic>=1`, `<model_type>=film_dot`

After training, models will be saved in the `saved_models_semantic` directory.

## Inference

Run the inference script based on your dataset type:

**For standard datasets:**
```bash
python inference_semantic.py \
  --dataset <dataset> --data_path <data_path> --model_path <model_path> \
  --use_semantic --model_type <model_type> --batch_size <batch_size> \
  --semantic_dim <semantic_dim> --cuda --gpu <gpu_id>
```

**For cold-start datasets:**
```bash
python inference_semantic.py \
  --dataset <dataset> --data_path <data_path> --model_path <model_path> \
  --use_semantic --model_type <model_type> --cold_start --batch_size <batch_size> \
  --semantic_dim <semantic_dim> --cuda --gpu <gpu_id>
```

## Citation

```bibtex
@inproceedings{wang2023diffrec,
  title = {Diffusion Recommender Model},
  author = {Wang, Wenjie and Xu, Yiyan and Feng, Fuli and Lin, Xinyu and He, Xiangnan and Chua, Tat-Seng},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages = {832–841},
  publisher = {ACM},
  year = {2023}
}
```
## Acknowledgment
LLM-DiffRec is developed based on the aforementioned paper and the original DiffRec codebase. The original implementation can be found at: https://github.com/YiyanXu/DiffRec. I sincerely thank the authors for their contribution to the community.

