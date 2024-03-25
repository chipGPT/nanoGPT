#!/bin/bash

# Define an array of method names
methods=("choice_one" "choice_two" "choice_three")

# Backup the original model.py file
cp model.py model_backup.py

# Loop through each method
for method in "${methods[@]}"; do
    cp model_backup.py model.py
    sed -n '138,142p' model.py
    sed -n '153,159p' model.py
    echo "Testing method: $method"

    # Restore the original model.py file
    cp model_backup.py model.py

    # Uncomment the lines for the current method and comment out the others
    if [ "$method" == "choice_one" ]; then
        pattern="#Gating_kv = nn.Linear(self.n_embd, self.kv_dim, bias=True, device=x.device)"
        file="model.py"
        # Find the line number where the pattern occurs
        line_num=$(grep -n "$pattern" "$file" | cut -d: -f1 | head -n 1)
        #echo $line_num
        if [ -n "$line_num" ]; then
            # Calculate the line range for sed
            start_line=$((line_num))
            end_line=$((line_num + 4))

            # Use sed to uncomment the specified line range
            echo $start_line
            echo $end_line
            sed -i "${start_line},${end_line}s/^ *#/                /" "$file"
        fi

        
        pattern="Gating_q = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)"
        file="model.py"
        line_num=$(grep -n "$pattern" "$file" | sed '1d' | cut -d: -f1 | head -n 1)
        echo $line_num
        if [ -n "$line_num" ]; then
            start_line=$line_num
            end_line=$((line_num + 6))
            sed -i "${start_line},${end_line}s/^                /                ##/" "$file"
        fi
        sed -n '138,142p' model.py
        sed -n '153,159p' model.py


    elif [ "$method" == "choice_two" ]; then
        pattern="#Gating_q = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)"
        file="model.py"
        # Find the line number where the pattern occurs
        line_num=$(grep -n "$pattern" "$file" | cut -d: -f1 | head -n 1)
        echo "notice here"
        echo $line_num
        if [ -n "$line_num" ]; then
            # Calculate the line range for sed
            start_line=$((line_num))
            end_line=$((line_num + 7))

            # Use sed to uncomment the specified line range
            echo $start_line
            echo $end_line
            sed -i "${start_line},${end_line}s/^ *#/                /" "$file"
            # comment out the other implementations
        fi


        pattern="Gating_q = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)"
        file="model.py"
        line_num=$(grep -n "$pattern" "$file" | sed '1d' | cut -d: -f1 | head -n 1)
        echo $line_num
        if [ -n "$line_num" ]; then
            start_line=$line_num
            end_line=$((line_num + 6))
            sed -i "${start_line},${end_line}s/^                /                ##/" "$file"
        fi

        sed -n '138,142p' model.py
        sed -n '153,159p' model.py
    elif [ "$method" == "choice_three" ]; then
        :
    fi

    # Run your training script with the modified model.py
    python3 data/shakespeare_char/prepare.py
    python3 train.py --out_dir=out --eval_interval=50 --log_interval=1 --device=cpu --block_size=2 --batch_size=2 --n_layer=2 --n_head=2 --n_embd=16 --lr_decay_iters=2 --gate --dtype="float32" --max_iter=100 --no-use_rotary_embeddings --use_abs_pos_embeddings --use_abs_pos_embeddings --no-use_post_ln

    python3 weight_vis.py --weight transformer.h.0.attn.c_attn_q.weight --graph matrix
    python3 weight_vis.py --weight transformer.h.0.attn.c_attn_k.weight --graph histogram
    python3 weight_vis.py --weight transformer.h.0.attn.c_attn_v.weight --graph matrix
    python3 weight_vis.py --weight transformer.h.0.attn.c_attn_q.weight --graph histogram
    python3 weight_vis.py --weight transformer.h.0.attn.c_attn_k.weight --graph matrix
    python3 weight_vis.py --weight transformer.h.0.attn.c_attn_v.weight --graph histogram
done

# Restore the original model.py file
cp model_backup.py model.py

# Remove the backup file
rm model_backup.py
