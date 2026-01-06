#!/bin/bash

for rar_file in *.rar; do
    if [ -f "$rar_file" ]; then
        echo "Decompressing: $rar_file"
        
        output_dir="${rar_file%.rar}"
        mkdir -p "$output_dir"
        
        unrar e "$rar_file" "$output_dir"
    fi
done

echo ""
echo "Decompress success!"