import os
from lvlm.LLaVA import LLaVA
import torch
import argparse
from tqdm import tqdm
import json 
import random
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from PIL import Image

LVLM_MAP = {
    'llava-1.5-13b-hf': LLaVA,
    'llava-1.5-7b-hf': LLaVA,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lvlm', type=str, default='llava-1.5-7b-hf')
    parser.add_argument('--dataset', type=str, default='MSCOCO')
    parser.add_argument('--inference_temp', type=float, default=0.1)
    parser.add_argument('--sampling_temp', type=float, default=1.0)
    parser.add_argument('--sampling_time', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--generate', type=bool, default=True)
    parser.add_argument('--num_data', type=int, default=5000)
    parser.add_argument('--image_layer', type=int, default=32)
    parser.add_argument('--text_layer', type=int, default=31)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--w', type=float, default=0.6)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def obtain_lvlm(args):
    lvlm_class = LVLM_MAP.get(args.lvlm)
    if not lvlm_class:
        raise ValueError(f"Unsupported LVLM: {args.lvlm}")

    return lvlm_class(args.lvlm)

def extract_tensors(data_dict):
    """Extracts tensors from a dictionary and converts them to a NumPy array."""
    tensor_list = []
    for obj, tensor_list_per_obj in data_dict.items():
        tensor_list.extend([t.cpu().numpy() for t in tensor_list_per_obj])  # Convert tensors to NumPy
    return np.array(tensor_list)  # Shape: (num_samples, 33)

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    fix_seed(0)
    args = parse_args()
    lvlm = obtain_lvlm(args)
    if args.dataset == "MSCOCO":
        MSCOCO_VAL_DIR = "MSCOCO_DATASET_PATH"
        COCO_ANNOTATION_PATH = "COCO_ANNOTATION_PATH"
          
    QUESTION = "Describe the given image in detail."
    
    # Load COCO annotations
    with open(COCO_ANNOTATION_PATH, "r") as f:
        coco_data = [json.loads(line) for line in f]
    
    coco_gt = random.sample(coco_data, args.num_data)
    
    global_cos_matrix_true_layer, global_cos_matrix_false_layer = [], []
  
    top_k_cos_matrix_true_layer, top_k_cos_matrix_false_layer = [], []

    if args.generate == True:
        for entry in tqdm(coco_gt, desc="Processing Images"):
            image_filename = entry["image"]  
            image_path = os.path.join(MSCOCO_VAL_DIR, image_filename)
    
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_filename} not found. Skipping.")
                continue
            
            image = Image.open(image_path).convert("RGB")
            
            # Perform inference using LLaVA
            result = lvlm.generate(image, QUESTION, entry["image_id"], args) 
  
            global_cos_matrix_true_flat = extract_tensors(result['global_cos_matrix_true'])
            global_cos_matrix_false_flat = extract_tensors(result['global_cos_matrix_false'])
         
            top_k_cos_matrix_false_flat = extract_tensors(result['top_k_cos_matrix_false'])
            top_k_cos_matrix_true_flat = extract_tensors(result['top_k_cos_matrix_true'])
        
            global_cos_matrix_true_layer.append(global_cos_matrix_true_flat)
            global_cos_matrix_false_layer.append(global_cos_matrix_false_flat)
         
            top_k_cos_matrix_true_layer.append(top_k_cos_matrix_true_flat)
            top_k_cos_matrix_false_layer.append(top_k_cos_matrix_false_flat)
     
        filtered_layers = lambda layers: [arr for arr in layers if arr.size > 0]
        stack_layers = lambda layers: np.vstack(filtered_layers(layers))
           
        global_cos_matrix_true_layer_stacked, global_cos_matrix_false_layer_stacked = stack_layers(global_cos_matrix_true_layer), stack_layers(global_cos_matrix_false_layer)
     
        top_k_cos_matrix_true_layer_stacked, top_k_cos_matrix_false_layer_stacked = stack_layers(top_k_cos_matrix_true_layer), stack_layers(top_k_cos_matrix_false_layer)

                
        def compute_layerwise_metrics(true_matrix, false_matrix, args):

            N, _, _ = true_matrix.shape
            M = false_matrix.shape[0]
            
            true_scores = true_matrix[:, args.text_layer, args.image_layer]
            false_scores = false_matrix[:, args.text_layer, args.image_layer]
      
            y_true = np.concatenate([np.ones(N), np.zeros(M)])
            y_scores = np.concatenate([true_scores, false_scores])

            auroc = roc_auc_score(y_true, y_scores)
            aupr = average_precision_score(y_true, y_scores)
            return {
                    'auroc': auroc,
                    'aupr': aupr,
                       }
                
        compute_layerwise_metrics(
                           args.w * global_cos_matrix_true_layer_stacked + (1 - args.w) *  top_k_cos_matrix_true_layer_stacked,
                           args.w * global_cos_matrix_false_layer_stacked + (1 - args.w) *  top_k_cos_matrix_false_layer_stacked,
                           args
                        )
         
   

        
if __name__ == "__main__":
    main()