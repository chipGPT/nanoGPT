#!/bin/bash

# Define an array of method names
methods=("choice_one" "choice_two" "choice_three")

# Backup the original model.py file
cp model.py model_backup.py

# Loop through each method
for method in "${methods[@]}"; do
    echo "Testing method: $method"

    # Restore the original model.py file
    cp model_backup.py model.py

    # Uncomment the lines for the current method and comment out the others
    if [ "$method" == "choice_one" ]; then
        pattern="#Gating_kv = nn.Linear(self.n_embd, self.kv_dim, bias=True, device=x.device)"
        file="model.py"

        # Find the line number where the pattern occurs
        line_num=$(grep -n "$pattern" "$file" | cut -d: -f1 | head -n 1)
        echo $line_num
        if [ -n "$line_num" ]; then
            # Calculate the line range for sed
            start_line=$((line_num))
            end_line=$((line_num + 4))

            # Use sed to uncomment the specified line range
            echo $start_line
            echo $end_line
            sed -i "${start_line},${end_line}s/^ *#/                /" "$file"
            # comment out the other implementations
            pattern="Gating_q = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)"
            file="model.py"
            line_num=$(grep -n "$pattern" "$file" | cut -d: -f1 | head -n 1)

            if [ -n "$line_num" ]; then
                start_line=$line_num
                end_line=$((line_num + 6))
                sed -i "${start_line},${end_line}s/^/##/" "$file"
            fi
        fi
        sed -n '138,142p' model.py
        sed -n '153,159p' model.py


    elif [ "$method" == "choice_two" ]; then
        sed -i 's/#Gating_q/Gating_q/' model.py
        sed -i 's/Gating_kv/#Gating_kv/' model.py
        sed -i 's/Gating_q_and_kv/#Gating_q_and_kv/' model.py
    elif [ "$method" == "choice_three" ]; then
        sed -i 's/#Gating_q_and_kv/Gating_q_and_kv/' model.py
        sed -i 's/Gating_kv/#Gating_kv/' model.py
        sed -i 's/Gating_q/#Gating_q/' model.py
    fi

    # Run your training script with the modified model.py
    python3 train.py

    # Optionally, you can save the output to a file specific to the method
    # python3 train.py [other arguments] > output_$method.txt
done

# Restore the original model.py file
cp model_backup.py model.py

# Remove the backup file
rm model_backup.py
