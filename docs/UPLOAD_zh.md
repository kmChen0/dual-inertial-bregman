# 上传前说明

上传整个 `dual-inertial-bregman` 文件夹内的内容，不要上传外层论文目录。README 是面向读者的英文运行说明。

已包含原论文 seed-1 实验代码、参考轨迹、四张图的重绘入口，以及新增 20-seed 实验的运行代码、汇总 CSV 和结果图。新增 NNLS/FISTA 等补充结果单独标注，不表示已经加入论文。约七百 MB 的新增原始运行记录没有打包，完整运行命令可以重新生成。

`results/generated/` 是本地检查产生的文件，已在 `.gitignore` 中排除，不应加入 Git。也不要加入 `.venv`、缓存或任何稿件源文件。作者暂未选定许可证，因此没有 LICENSE 文件。若需要 DOI、arXiv 编号或 GitHub 地址，请在核实后补入 CITATION.cff 和 README，当前没有填写占位链接。

建议 GitHub 仓库名：`dual-inertial-bregman`。新建空仓库后，可以在此文件夹单独初始化 Git，再按 GitHub 页面提示推送。不要在原论文总目录执行 `git add .`。

本次只准备本地目录与压缩包，没有创建远程仓库，没有上传，也没有提交原工作区的 Git 变更。
