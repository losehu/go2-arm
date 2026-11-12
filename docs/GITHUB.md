# 上传已有 GitHub 仓库

本目录是独立 Git 项目，包含训练所需资产；无需 Git LFS 才能读取本快照。权重、日志、SSH 连接与本机预览运行目录已忽略。

先在 GitHub 创建一个空仓库，然后在项目目录执行，将 `OWNER/REPOSITORY` 换成目标：

```bash
git status
git remote add origin git@github.com:OWNER/REPOSITORY.git
git push -u origin main
```

如果使用旁边的源码压缩包，它不包含 `.git` 历史。解压后先执行以下步骤，再配置远端：

```bash
git init -b main
git add .
git commit -m "Import v11 training and full assets"
```

如果已配置 `origin`，先用 `git remote -v` 核对目标；需要修改时使用 `git remote set-url origin ...`。本次 v11 更新的发布目标为 `losehu/go2-arm`；后续推送前核对当前目录的远端配置。

资产原始条款见 `THIRD_PARTY.md`，其中相机支架来源标为 Proprietary。完整本地快照保留了它们；公开仓库的再分发授权需要由发布者确认。仓库没有把这些资产重新许可为 MIT / Apache。
