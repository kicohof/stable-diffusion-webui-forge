@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=

@REM Uncomment following code to reference an existing A1111 checkout.
@REM set A1111_HOME=Your A1111 checkout dir
@REM
@REM set VENV_DIR=%A1111_HOME%/venv
@REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% ^
@REM  --ckpt-dir %A1111_HOME%/models/Stable-diffusion ^
@REM  --hypernetwork-dir %A1111_HOME%/models/hypernetworks ^
@REM  --embeddings-dir %A1111_HOME%/embeddings ^
@REM  --lora-dir %A1111_HOME%/models/Lora

set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% ^
 --xformers ^
 --ckpt-dir 'D:\ai\text-to-image\models\checkpoints\' ^
 --hypernetwork-dir 'D:\ai\text-to-image\models\hypernetworks' ^
 --embeddings-dir 'D:\ai\text-to-image\models\embeddings\' ^
 --lora-dir 'D:\ai\text-to-image\models\loras\' ^
 --vae-dir 'D:\ai\text-to-image\models\vae\' ^
 --esrgan-models-path 'D:\ai\text-to-image\models\upscale_models\' ^
 --cuda-malloc ^
 --cuda-stream ^
 --pin-shared-memory

echo %COMMANDLINE_ARGS%

call webui.bat
