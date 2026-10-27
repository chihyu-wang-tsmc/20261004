# VerilogEval

給 `deep20.py` 的 `VerilogEval` 用的原始資料：[NVlabs/verilog-eval](https://github.com/NVlabs/verilog-eval)
v2 的 `dataset_spec-to-rtl/` 裡 156 個 `*_prompt.txt`，**沒有改過任何內容**。放進 repo 之後就不用另外下載。

## 用法

```bash
cp -r VerilogEval ~/code_benchmarks/
```

`deep20.py` 讀的是 `<CODE>/VerilogEval/dataset_spec-to-rtl/*_prompt.txt`。`<CODE>` 是
`~/chihyu-wang-tsmc-datasets/code-benchmarks`（存在的話），否則是 `~/code_benchmarks`。
放好之後 `python deep20_coding_langs.py check` 應該看到 `VerilogEval：156 題`。

## 來源

- 上游 `main` @ `c498220`（2025-02-07），取自完整快照 repo `chihyu-wang-tsmc/VerilogEval`。
- 授權是 MIT，`LICENSE` 是上游原檔。

## 為什麼只放 prompt

完整的 `chihyu-wang-tsmc/VerilogEval` 在公司 clone 下來是空資料夾，刪掉 build 腳本和 Dockerfile 之後也一樣。
`deep20.py` 只讀 prompt，所以這裡只放 prompt：同一題的 `_ref.sv`、`_test.sv`（參考解答和 testbench）、
`dataset_code-complete-iccad2023/`、`v1/` 都沒有放。要跑官方評測的話去上游拿。
