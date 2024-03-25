#!/bin/bash

# Define an array of method names
methods=("Gating_kv" "Gating_q" "Gating_q_and_kv")

# Backup the original model.py file
cp model.py model_backup.py

# Loop through each method
for method in "${methods[@]}"; do
    echo "Testing method: $method"

    # Restore the original model.py file
    cp model_backup.py model.py

    # Uncomment the lines for the current method and comment out the others
    if [ "$method" == "Gating_kv" ]; then
        sed -i 's/#Gating_kv/Gating_kv/' model.py
        sed -i 's/Gating_q/#Gating_q/' model.py
        sed -i 's/Gating_q_and_kv/#Gating_q_and_kv/' model.py
    elif [ "$method" == "Gating_q" ]; then
        sed -i 's/#Gating_q/Gating_q/' model.py
        sed -i 's/Gating_kv/#Gating_kv/' model.py
        sed -i 's/Gating_q_and_kv/#Gating_q_and_kv/' model.py
    elif [ "$method" == "Gating_q_and_kv" ]; then
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
