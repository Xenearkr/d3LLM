# 将 d3LLM 推到 GitHub 并与同学协作

## 当前状态（简要）

- 本地已有 git，远程为 `origin` → `https://github.com/hao-ai-lab/d3LLM.git`
- 当前分支 `main`，比 `origin/main` 多 2 个 commit，另有未提交修改与未跟踪文件

---

## 一、把本地改动推到 GitHub

### 1. 提交本地修改（可选，按需）

```bash
cd /home/u-liujc/Codes/d3LLM

# 查看状态
git status

# 添加要提交的文件（示例：评估脚本与文档）
git add eval_scripts/run_code_eval.sh eval_scripts/README.md eval_scripts/*.md
git add utils/utils_LLaDA/postprocess_code_humaneval.py

# 提交
git commit -m "eval: add run_code_eval.sh, HumanEval/MBPP docs, LLaDA postprocess fix"
```

### 2. 推送到远程

- **若你有 `hao-ai-lab/d3LLM` 的写权限**（成员或自己建的 org 仓库）：

```bash
git push origin main
```

- **若没有写权限**：需要先在自己账号下“Fork”该仓库，再把远程改成你的 fork，再 push：

```bash
# 在 GitHub 网页上 Fork hao-ai-lab/d3LLM → 得到 https://github.com/<你的用户名>/d3LLM

# 添加你的 fork 为远程（保留原 origin 方便同步上游）
git remote add myfork https://github.com/<你的用户名>/d3LLM.git

# 推送到你的 fork
git push myfork main
```

之后协作可以以“你的 fork”为共享仓库（见下）。

---

## 二、与同学协作的两种常见方式

### 方式 A：在“一个共享仓库”上协作（你有该仓库写权限）

1. **GitHub 仓库设置**  
   仓库 → **Settings** → **Collaborators** → **Add people**，添加同学的 GitHub 账号。
2. **同学克隆**  
   同学执行：`git clone https://github.com/<组织或你>/d3LLM.git`，之后在同一仓库里 `git pull` / `git push`（或走分支，见下）。

### 方式 B：Fork + 分支协作（原仓库你无写权限）

1. 你 Fork 后推送到 `myfork`（如上），把 fork 的仓库链接发给同学。
2. 同学 Fork 你的仓库（或同一 org 下的 d3LLM），或你把他们加为你 fork 的 Collaborators。
3. 约定分支策略，例如：
   - `main`：稳定可跑版本；
   - 每人/每任务一个分支，如 `dev/zhangsan`、`feat/eval-mbpp`，开发完再提 PR 到 `main`。

---

## 三、常用协作命令（同一仓库或 fork）

```bash
# 拉取最新
git pull origin main

# 在分支上开发
git checkout -b dev/xxx
# 改完
git add ...
git commit -m "..."
git push origin dev/xxx
# 然后在 GitHub 上开 Pull Request 到 main
```

---

## 四、注意事项

- **大文件/结果不要提交**：`.gitignore` 已包含 `evals_results/`、`eval_tmp/`、`output_model/` 等，勿强制添加。
- **敏感信息**：不要把密钥、密码写进仓库；可用环境变量或本地配置文件（且该文件在 .gitignore 中）。
- **冲突**：多人改同一文件时先 `git pull` 再改，冲突时手工解决冲突后再 commit、push。

若你确定使用“自己的 fork”还是“hao-ai-lab 下的仓库”作为主远程，可把选择告诉我，我可以按你的选择把上面步骤精简成你专属的一步步清单（含你要执行的命令）。
