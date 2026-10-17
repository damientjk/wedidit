# SETUP.md: Connect to the team LinuxONE server

Get these privately from Damien (never put them in the repo): SSH key `datathon.pem`, server IP `<SERVER_IP>`, username `linux1`, Jupyter token.

## 1. Clone the repo
```bash
git clone https://github.com/damientjk/wedidit.git
cd wedidit
```

## 2. Prepare the key

**Windows (PowerShell)**
```powershell
icacls "C:\Users\YOURNAME\Downloads\datathon.pem" /inheritance:r
icacls "C:\Users\YOURNAME\Downloads\datathon.pem" /grant:r "$($env:USERNAME):(R)"
```

**Mac/Linux**
```bash
chmod 600 ~/Downloads/datathon.pem
```

## 3. Connect

**Windows**
```powershell
ssh -i "C:\Users\YOURNAME\Downloads\datathon.pem" linux1@<SERVER_IP>
```

**Mac/Linux**
```bash
ssh -i ~/Downloads/datathon.pem linux1@<SERVER_IP>
```
Type `yes` at the first fingerprint prompt.

## 4. Jupyter Lab (Ethan's labs)
```powershell
ssh -i "C:\Users\YOURNAME\Downloads\datathon.pem" -L 38888:localhost:38888 linux1@<SERVER_IP>
```
Keep this terminal open, then open http://localhost:38888 and enter the Jupyter token. Closing the terminal kills the tunnel.

## 5. Edit server files from VS Code (optional)
VS Code Remote-SSH does not work on this server: it needs a VS Code server build for the s390x architecture, which does not exist. Use the **SSH FS** extension instead:
1. Install **SSH FS** (by Kelvin Schoutsen).
2. Create a configuration: host `<SERVER_IP>`, port `22`, username `linux1`, private key = your `datathon.pem`, root `/home/linux1/wedidit`.
3. Right-click the configuration and choose **Add as Workspace folder**. Files save straight to the server.
4. For commands, open a normal terminal and `ssh` in.

## 6. First checks on the server
```bash
uname -m                 # expect s390x
python3 --version
docker ps
mkdir -p ~/yourname      # your own working folder
pip install xgboost shap fastapi uvicorn
```
If any install fails, tell Damien immediately.

## 7. Rules on the shared account
- Everyone is `linux1`. Work only inside `~/yourname/`.
- Git is the only shared space. Pull before push, commit small.
- Never commit `*.pem`, `.env`, tokens, or the server IP. Add them to `.gitignore`.
