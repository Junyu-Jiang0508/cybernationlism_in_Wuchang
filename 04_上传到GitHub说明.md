# 上传到 GitHub 的步骤

## 一、在本地初始化 Git 并提交

在 **PowerShell** 或 **命令提示符** 中，进入项目目录后依次执行：

```powershell
# 1. 进入项目目录
cd "C:\Users\Jain Farstrider\Desktop\COG181\Final_Project"

# 2. 初始化 Git 仓库
git init

# 3. 添加所有文件（.gitignore 已配置，venv、data 等不会上传）
git add .

# 4. 第一次提交
git commit -m "Initial commit: 知乎爬虫配置与运行脚本"
```

---

## 二、在 GitHub 上创建新仓库

1. 打开 **https://github.com** 并登录。
2. 点击右上角 **“+”** → **“New repository”**。
3. 填写：
   - **Repository name**：例如 `COG181-Final-Project` 或 `zhihu-crawler`
   - **Description**（可选）：例如「知乎关键词爬虫项目」
   - 选择 **Public**。
   - **不要**勾选 “Add a README file”（本地已有代码）。
4. 点击 **“Create repository”**。

创建完成后，页面上会显示仓库地址，形如：  
`https://github.com/你的用户名/仓库名.git`

---

## 三、关联远程仓库并推送

在**同一目录**下执行（把下面的地址换成你自己的仓库地址）：

```powershell
# 1. 添加远程仓库（替换为你的 GitHub 仓库地址）
git remote add origin https://github.com/你的用户名/仓库名.git

# 2. 推送到 GitHub（首次推送并设置上游分支）
git branch -M main
git push -u origin main
```

如果 GitHub 要求登录，会弹出浏览器或提示输入用户名和**个人访问令牌（Token）**（密码已不再支持，需在 GitHub → Settings → Developer settings → Personal access tokens 创建）。

---

## 四、之后有修改时如何更新

```powershell
cd "C:\Users\Jain Farstrider\Desktop\COG181\Final_Project"
git add .
git commit -m "描述你的修改"
git push
```

---

## 五、当前 .gitignore 已忽略的内容

以下内容**不会**被提交到 GitHub：

- `venv/` 虚拟环境
- `.env` 环境变量与密钥
- `data/` 爬虫生成的数据
- `*_user_data_dir/` 登录缓存
- `__pycache__/`、`.idea/`、`.vscode/` 等

如需调整忽略规则，可编辑项目根目录下的 **`.gitignore`** 文件。
