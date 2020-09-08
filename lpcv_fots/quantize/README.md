
## Quantize Model

After we finetuned, pruned fots model, the final step is to quantize it. For quantization details, please check ```quantize_model.py```. Run the command:

```sh
python3 quantize_model.py --calibrate-folder /path/to/calibrate/dir --pretrain-model /path/to/pretrain/model \
--backends fbgemm --save-dir /path/to/save/dir
```

