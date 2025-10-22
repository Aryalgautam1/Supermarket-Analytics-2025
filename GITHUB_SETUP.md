# 📘 GitHub Setup Guide

Quick guide to push your code to GitHub for deployment.

## 🎯 **Prerequisites**

- Git installed on your computer
- GitHub account (create at [github.com](https://github.com))

## 📝 **Step-by-Step Instructions**

### **1. Create GitHub Repository**

1. Go to [github.com](https://github.com)
2. Click the "+" icon → "New repository"
3. Fill in details:
   - **Repository name:** `supermarket-analytics`
   - **Description:** "AI-powered supermarket analytics platform with sales forecasting, inventory management, and price optimization"
   - **Visibility:** Public (required for free deployment)
   - **Don't** initialize with README (we already have one)
4. Click "Create repository"

### **2. Initialize Git Locally**

Open terminal in your `Supermarket-Analytics` folder:

```bash
# Navigate to your project
cd "Supermarket-Analytics"

# Initialize Git
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Supermarket Analytics app"
```

### **3. Connect to GitHub**

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/supermarket-analytics.git

# Push code
git branch -M main
git push -u origin main
```

### **4. Verify Upload**

- Go to your GitHub repository
- You should see all your files
- Verify `data/SuperMarket Analysis.csv` is there

---

## 🔄 **Updating Code Later**

After making changes:

```bash
# Check what changed
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of what you changed"

# Push to GitHub
git push
```

---

## 🔐 **Using GitHub Desktop (Alternative)**

Prefer GUI? Use GitHub Desktop:

1. Download from [desktop.github.com](https://desktop.github.com)
2. Sign in with GitHub account
3. Add local repository (your Supermarket-Analytics folder)
4. Commit changes
5. Push to GitHub

---

## ⚠️ **Common Issues**

### Issue: "Permission denied"
**Solution:** Use HTTPS URL or set up SSH keys

### Issue: "File too large"
**Solution:** GitHub has 100MB file limit
- If your CSV is >100MB, use Git LFS or host data elsewhere

### Issue: "Remote already exists"
**Solution:**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/supermarket-analytics.git
```

---

## 📦 **What Gets Uploaded**

✅ **Included:**
- All Python files (.py)
- requirements.txt
- Data files (.csv)
- Configuration files (.toml)
- Documentation (.md)

❌ **Excluded (via .gitignore):**
- `__pycache__/`
- `.env` files
- `downloads/` generated files
- `models/` trained models (too large)
- `github_drop/` old files

---

## 🎓 **Git Commands Cheat Sheet**

```bash
# Check status
git status

# Add all changes
git add .

# Add specific file
git add filename.py

# Commit changes
git commit -m "Your message"

# Push to GitHub
git push

# Pull latest changes
git pull

# View commit history
git log

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Create new branch
git checkout -b new-feature

# Switch branch
git checkout main

# See all branches
git branch
```

---

## 🌟 **Best Practices**

1. **Commit often** - Small, frequent commits are better
2. **Write clear messages** - Describe what changed and why
3. **Pull before push** - Get latest changes first
4. **Use branches** - For new features or experiments
5. **Review before commit** - Check `git status` and `git diff`

---

## 📋 **Commit Message Examples**

Good commit messages:
```
✅ "Add caching to improve performance"
✅ "Fix: Resolve data loading error"
✅ "Update README with deployment instructions"
✅ "Feat: Add XGBoost model to forecaster"
```

Bad commit messages:
```
❌ "Update"
❌ "Fix stuff"
❌ "Changes"
❌ "asdfasdf"
```

---

## 🚀 **Ready for Deployment!**

Once your code is on GitHub:

1. ✅ Code is backed up
2. ✅ Version controlled
3. ✅ Ready for deployment
4. ✅ Can collaborate with others

**Next step:** Follow `DEPLOYMENT.md` to deploy your app!

---

## 🆘 **Need Help?**

- **Git Documentation:** https://git-scm.com/doc
- **GitHub Guides:** https://guides.github.com
- **Learn Git:** https://learngitbranching.js.org

---

**Happy coding! 💻**

