import torch.utils.checkpoint
import torch
import argparse
import json
from tqdm import tqdm 
from transformer import VoltronTransformerPretrained, TokenizeMask


def get_model(demo_type = 'defects4j', pretrain_type = '6B'):
    num_layer = 2
    target_dim = 512
    if demo_type == 'defects4j' and pretrain_type == "16B":
        target_dim = 1024
    if pretrain_type == '16B':
        dim_model = 6144
    elif pretrain_type == '6B':
        dim_model = 4096
    elif pretrain_type == '350M':
        dim_model = 1024
    
    if target_dim == 1024:
        num_head = 16
    elif target_dim == 512:
        num_head = 8
    elif target_dim == 256:
        num_head = 4

    model = VoltronTransformerPretrained(
        num_layer=num_layer, dim_model=dim_model, num_head=num_head, target_dim=target_dim
    )
    model.load_state_dict(torch.load(
        f'model_checkpoints/{demo_type}_{pretrain_type}'), strict=False)
    model.to("cuda")
    model.eval()
    return model


def buglines_prediction(model, code_content, demo_type = 'defects4j', pretrain_type = '6B'):
    tokenize_mask = TokenizeMask(pretrain_type)
    code_file = code_content.split("\n")
    filtered_to_orig_mapping = {}
    filtered_code = []
    curr_orig = 0
    curr_filtered = 0
    for code_line in code_file:
        code_line = code_line + "\n"
        if code_line and not code_line.strip().startswith('/') and not code_line.strip().startswith('*') and not code_line.strip().startswith('#') and not code_line.strip() == '{' and not code_line.strip() == '}' and code_line not in filtered_code:
            if len(code_line.strip()) > 0:
                filtered_code.append(code_line)
                filtered_to_orig_mapping[curr_orig] = curr_filtered
                curr_filtered += 1

        curr_orig += 1

    code_lines = ''.join(filtered_code)
    input, mask, input_size, decoded_input = tokenize_mask.generate_token_mask(
        code_lines)
    input = input[None, :]
    mask = mask[None, :]
    predictions = model(input, mask)
    probabilities = torch.flatten(torch.sigmoid(predictions))
    real_indices = torch.flatten(mask == 1)            
    probabilities = probabilities[real_indices].tolist()        
    decoded_input_list = decoded_input.split('\n')
    decoded_input = [line.lstrip('\t')
                        for line in decoded_input_list]
    decoded_input = "\n".join(decoded_input)
    probabilities = probabilities[:input_size+1]
    print(probabilities)
    most_sus = list(
        map(lambda x: 1 if x > 0 else 0, probabilities))
    result_dict = []
    for i, p in enumerate(most_sus):
        if p == 1 and len(filtered_code[i].strip()) > 1:
            ind = i-1 if demo_type == "defects4j" else i
            result_dict.append({"line": ind, "score": round(probabilities[i]*100,2)})

    return result_dict

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Path to JSON dataset")
    ap.add_argument("--output", help="Path to JSON dataset")

    args = ap.parse_args()
    

    with open(args.input, 'r') as f:
        json_data = [json.loads(l) for l in f]

    final_d4j_fl = {}

    model = get_model()

    for item in tqdm(json_data):
        final_d4j_fl[item["file_name"]] = buglines_prediction(model, item["text"])
    

    with open(args.output, 'w') as f:
        json.dump(final_d4j_fl, f)