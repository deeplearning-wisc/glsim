import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import nltk
import pickle
from pattern.en import singularize


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

class LLaVA:

    def __init__(self, version):
        self.version = version
        self.build_model()

    def build_model(self):
        model_name = f"llava-hf/{self.version}"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            output_attentions=True,
            output_hidden_states=True,
            attn_implementation="eager",
        ).to(0)
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.im_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        self.im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        CACHE_PATH = "CHAIR_CACHE_PATH"
        self.evaluator = pickle.load(open(CACHE_PATH, 'rb'))

    def generate(self, image, question, img_id, args):
        
        temp = args.inference_temp
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": question
                                    },
                                    {
                                        "type": "image"
                                    }
                                ]
                            }
                        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
        
        input_ids = inputs['input_ids'][0] 
        
        
        image_start = (input_ids == self.image_token_id).nonzero(as_tuple=True)[0][0].item()
        image_end =  (input_ids == self.image_token_id).nonzero(as_tuple=True)[0][-1].item()
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=temp,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        output_ids = output.sequences[0]
        final_ans = self.processor.decode(output_ids, skip_special_tokens=True).split('ASSISTANT: ')[-1].strip()
        # Identify answer token range
        answer_start = input_ids.shape[0]
        hidden_states = output.hidden_states  # Tuple: (num_layers, batch, seq_len, hidden_dim)


        tokens = nltk.word_tokenize(final_ans.lower())
        if len(tokens) != len([singularize(w) for w in tokens]):
            print("warning, the token number unmatched")
        
        # Internal Confidence
        hidden_states = torch.stack(output.hidden_states[0])
        question_hidden_states = hidden_states[:, 0, -1, :] 
    
        
        with torch.inference_mode():
            curr_layer_logits = self.model.language_model.lm_head(hidden_states).cpu().float()
            logit_scores = torch.nn.functional.log_softmax(curr_layer_logits, dim=-1)
            softmax_probs_raw = torch.nn.functional.softmax(logit_scores, dim=-1)
            
        softmax_probs_raw = softmax_probs_raw.detach().cpu().numpy()
        
        softmax_probs_raw = softmax_probs_raw[
            :, :, image_start : image_end + 1
        ]
                
        img_repr = torch.stack(output.hidden_states[0]).squeeze(1).squeeze(1)
        
        img_repr = img_repr[:, image_start : image_end + 1]

        softmax_probs_raw = softmax_probs_raw.transpose(3, 0, 2, 1)

        softmax_probs_raw = softmax_probs_raw.max(axis=3)
        
        global_cos_matrix_true, global_cos_matrix_false, top_k_cos_matrix_true, top_k_cos_matrix_false =  {}, {}, {}, {}
    
        output_ids_rs = output_ids[answer_start:]
        cap_dict = self.evaluator.compute_hallucinations(img_id,final_ans)

        num_layers = img_repr.shape[0]

        k = args.k
        
        with torch.no_grad():
            for true_idx in cap_dict['recall_idxs']:
                inputs = self.processor.tokenizer(tokens[true_idx], return_tensors="pt", add_special_tokens=False).input_ids
                if tokens[true_idx] not in global_cos_matrix_true.keys():
        
                    global_cos_matrix_true[tokens[true_idx]] = list()
                    top_k_cos_matrix_true[tokens[true_idx]] = list()
             
                    token = inputs[0, 0]
                    if  torch.where(output_ids_rs == token)[0].numel() != 0:
                            
                            toke_idx = torch.where(output_ids_rs == token)[0][0]
                            token_rep = torch.stack(output.hidden_states[toke_idx]).squeeze(1).squeeze(1)

                            token_rep_norm = F.normalize(token_rep, p=2, dim=-1)  
                            global_emb_norm = F.normalize(question_hidden_states, p=2, dim=-1)
                            global_cos_matrix = torch.matmul(global_emb_norm, token_rep_norm.T) 
                            global_cos_matrix_true[tokens[true_idx]].append(global_cos_matrix)
                            
                            token_prob = softmax_probs_raw[token]
                            top_k_indices = np.argsort(token_prob, axis=1)[:, -k:]  # Select last k (highest probabilities)
                            top_k_embeddings = F.normalize(img_repr[torch.arange(num_layers).unsqueeze(1), top_k_indices, :] ,p=2, dim=-1)

                            cos_sim_k = torch.einsum('id,jkd->ijk', token_rep_norm, top_k_embeddings)
                            top_k_cos_matrix = cos_sim_k.mean(dim=-1)
                            top_k_cos_matrix_true[tokens[true_idx]].append(top_k_cos_matrix)
                                                                         
            for false_idx in cap_dict['hallucination_idxs']:
                inputs = self.processor.tokenizer(tokens[false_idx], return_tensors="pt", add_special_tokens=False).input_ids
                if tokens[false_idx] not in global_cos_matrix_false.keys():
                  
                    global_cos_matrix_false[tokens[false_idx]] = list()
                    top_k_cos_matrix_false[tokens[false_idx]] = list()
        
                    token = inputs[0, 0]
                    if  torch.where(output_ids_rs == token)[0].numel() != 0:
                            toke_idx = torch.where(output_ids_rs == token)[0][0]
                            token_rep = torch.stack(output.hidden_states[toke_idx]).squeeze(1).squeeze(1)
                            token_prob = softmax_probs_raw[token]

                            token_rep_norm = F.normalize(token_rep, p=2, dim=1)  
                            global_emb_norm = F.normalize(question_hidden_states, p=2, dim=1)
                            global_cos_matrix = torch.matmul(global_emb_norm, token_rep_norm.T) 
                            global_cos_matrix_false[tokens[false_idx]].append(global_cos_matrix)
                            
                            token_prob = softmax_probs_raw[token]
                            top_k_indices = np.argsort(token_prob, axis=1)[:, -k:]  # Select last k (highest probabilities)
                            top_k_embeddings = F.normalize(img_repr[torch.arange(num_layers).unsqueeze(1), top_k_indices, :] ,p=2, dim=-1)
                     
                            cos_sim_k = torch.einsum('id,jkd->ijk', token_rep_norm, top_k_embeddings)
                            top_k_cos_matrix = cos_sim_k.mean(dim=-1)
                            top_k_cos_matrix_false[tokens[false_idx]].append(top_k_cos_matrix)
           
        result = {
            "global_cos_matrix_true": global_cos_matrix_true,
            "global_cos_matrix_false": global_cos_matrix_false, 
            "top_k_cos_matrix_true": top_k_cos_matrix_true,
            "top_k_cos_matrix_false": top_k_cos_matrix_false,
        }
    
        return result
    
