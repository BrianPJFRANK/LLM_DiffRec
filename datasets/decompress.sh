#!/bin/bash

for gz_file in compressed_file/*.gz; do

    if [ -f "$gz_file" ]; then
        
        base_name=$(basename "$gz_file")
        
        pure_name="${base_name%.gz}"
        
        target_dir="raw_$pure_name"
        
        echo "Processing: $base_name -> directory: $target_dir"
        
        mkdir -p "$target_dir"
        
        gunzip -c "$gz_file" > "$target_dir/$pure_name"
        
    fi
done

echo "Decompression Finished!"
